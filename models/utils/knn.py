# Torch implementation of finding k nearest neighbors

import torch

def knn_torch(points_from: torch.Tensor, points_to: torch.Tensor, k_neighbors: int=2, spatial_dimension: int=2, use_indices: torch.Tensor=None, bound: float=32.0, padding: bool=True) -> torch.Tensor:
    '''
    Input: points_from to find the k-nearest neighbors to in points_to

    use_indices: select the k nearest neighbors yourself

    use cyclic padding

    Outputs 2 tensors - Distances and the edge index(flattened)
    '''
    # num generate special
    points = points_to
    s = points.size()
    if padding:
        points = torch.reshape(points, (s[0], s[1], 1))
        s = points.size()
        l = [points_to]
        l.append(torch.reshape(torch.cat((points[:,0], points[:,1] + bound), dim=1), (s[0], s[1]))) # move all points up
        l.append(torch.reshape(torch.cat((points[:,0], points[:,1] - bound), dim=1), (s[0], s[1]))) # move all points down
        l.append(torch.reshape(torch.cat((points[:,0]+ bound, points[:,1]), dim=1), (s[0], s[1]))) # move all points right
        l.append(torch.reshape(torch.cat((points[:,0]- bound, points[:,1]), dim=1), (s[0], s[1]))) # move all points left
        l.append(torch.reshape(torch.cat((points[:,0]+ bound, points[:,1]+ bound), dim=1), (s[0], s[1]))) # move all points right, up
        l.append(torch.reshape(torch.cat((points[:,0]+ bound, points[:,1]- bound), dim=1), (s[0], s[1]))) # move all points right, down
        l.append(torch.reshape(torch.cat((points[:,0]- bound, points[:,1]+ bound), dim=1), (s[0], s[1]))) # move all points left, up
        l.append(torch.reshape(torch.cat((points[:,0]- bound, points[:,1]- bound), dim=1), (s[0], s[1]))) # move all points left, down
        p_all = torch.stack(l, dim=0)
        points_to = torch.reshape(p_all, (9*s[0], 2))

    distances = torch.cdist(points_from, points_to, p=spatial_dimension)
    # Compute Edge Index
    dist, edge_index = torch.sort(distances, dim=1)
    num = torch.tensor([])
    if use_indices != None:
        num = use_indices
    else:
        num = torch.arange(1, k_neighbors+1)
    edge_index = torch.index_select(edge_index, 1, num)
    edge_index = edge_index.flatten()
    points_n = torch.index_select(points_to, 0, edge_index)
    p_to = torch.repeat_interleave(points_from, k_neighbors, dim=0)
    dis = points_n - p_to
    return dis, edge_index%s[0] # dist (originally)
