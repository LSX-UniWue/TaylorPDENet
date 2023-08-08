import torch
from torch import nn
from . import derivative as d
from .utils import plot

class TP_layer(nn.Module):
    def __init__(self,
                 N: int,
                 num_neighbors: int,
                 TP_order: int,
                 spatial_dimension: int=2,
                 point_wise: bool=False,
                 plot_deriv: bool=False):
        """
        TP-Layer uses the Taylorpolynomial to approximate the partial derivatives for some points at time t 
        and uses these to cumpute the function values at t+1
        """
        super().__init__()
        self.N = N # number of points
        self.size = N*N
        self.dim = spatial_dimension
        self.k = num_neighbors
        self.TP_order = TP_order
        self.point_wise = point_wise
        self.plot_deriv = plot_deriv
        sum = 0
        for x in range(TP_order):
            sum = sum+x+1
        self.num_derivatives = sum+TP_order
        if point_wise:
            self.linear = PDELinear(N*N, self.num_derivatives)
        else:
            self.linear = PDELinear(1, self.num_derivatives)
            # uncomment if u00 should have own weight
            self.w00 = 0 # nn.Parameter(torch.tensor(0, dtype=torch.float32))

    def forward(self,
                x: torch.Tensor,
                points: torch.Tensor,
                edge_index: torch.Tensor,
                dt: float,
                dist: torch.Tensor,
                ) -> torch.Tensor:
        """
        Input: Vertices(Coordinates and Features) flattened, List of neighbors for vertices -> flattened

        Output: Forecasted points: U_(t+1) = U(t) + dt * dU/dt
        """
        #with torch.no_grad():
        batch = x.size()[0]
        # [batch, points, dim]
        points = torch.reshape(points, (batch, self.size, self.dim))
        values = torch.reshape(x, (batch, self.N*self.N, 1))
        knn_points = torch.zeros(batch, self.size*self.k, self.dim, device=x.device)
        knn_features = torch.zeros(batch, self.size*self.k, 1, device=x.device)
        for i in range(batch):
            p = points[i]
            v = values[i]
            e = edge_index[i]
            knn_points[i] = torch.index_select(p, 0, e)
            knn_features[i] = torch.index_select(v, 0, e)
        # k nearest neighbors points
        points = torch.reshape(points, (batch*self.size, self.dim))
        knn_points = torch.reshape(knn_points, (batch*self.size, self.k, self.dim))
        # k nearest neighbors features
        knn_features = torch.reshape(knn_features, (batch*self.size, self.k))
        # compute derivatives for each vertex
        with torch.no_grad():
            derivatives = d.solveDerivatives(vertices=points,
                                            neighbors=knn_points,
                                            vertices_values=x.flatten(),
                                            neighbors_values=knn_features,
                                            num_derivatives=self.num_derivatives,
                                            knn=self.k,
                                            order=self.TP_order,
                                            dist=dist)
        # derivatives containes for each vertex its derivatives up to the given order
        derivatives = torch.reshape(derivatives, (batch, self.size, self.num_derivatives))
        du = self.linear(derivatives)
        du = torch.sum(du, dim=-1)
        result = x + dt*du # x*self.w00
        return result


class PDELinear(nn.Module):
  def __init__(self, dim1, dim2, dtype: torch.dtype=torch.float32):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros((1, dim1, dim2), dtype=dtype))
    #self.weight = nn.Parameter(torch.tensor([[[1.0, 0.7, 1.5, 1.0, 0.5]]]))
    #self.weight = nn.Parameter(torch.tensor([[[0.3, 0.0, 0.4, 0.0, 0.0]]]))
    lim = 0.1  # initialize weights and bias
    nn.init.uniform_(self.weight, -lim, +lim)

  def forward(self, x):
    r = x * self.weight
    return r