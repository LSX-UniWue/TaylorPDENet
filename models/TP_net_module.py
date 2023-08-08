import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn.functional import mse_loss
from typing import Any

from . import TP_layer as tp
from .utils import plot


class TPNet(nn.Module):
    def __init__(self,
                 N: int=0,
                 TP_order: int=1,
                 num_neighbors: int=1,
                 point_wise: bool=False,
                 plot_deriv: bool=False) -> torch.Tensor:
        """
        """
        super().__init__()
        sum = 0
        for x in range(TP_order):
            sum = sum+x+1
        self.num_derivatives = sum+TP_order
        self.N = N
        self.size = N*N
        # append dt-blocks
        self.dt_block = tp.TP_layer(N=N,
                                    num_neighbors=num_neighbors,
                                    TP_order=TP_order,
                                    point_wise=point_wise,
                                    plot_deriv=plot_deriv)

    def forward(self, x, points, edge_index, dt: float, num_blocks: int, dist: torch.Tensor):
        list = []
        # list dt blocks result und torch.stack
        for i in range(num_blocks):
            x = self.dt_block(x=x, points=points, edge_index=edge_index, dt=dt, dist=dist)
            list.append(x)
        tot = torch.stack(list, dim=1)
        return tot

class TPNetModule(pl.LightningModule):
    def __init__(self, net: nn.Module,
                 learning_rate: float,
                 weight_decay: float,
                 plot: bool=False,
                 point_wise: bool=False,
                 order: int=2,
                 N: int=64,
                 equation: str='',
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.net = net
        self.loss = mse_loss
        self.lr = learning_rate
        self.wd = weight_decay
        self.order = order
        self.plot = plot
        self.N = N
        self.point_wise = point_wise
        self.equation = equation
        self.weight_dict = []
        for i in range(order+1):
            for j in range(order+1-i):
                self.weight_dict.append(f'u{i}{j}')
        

    def step(self, batch: Any):
        x, y, points, edge_index, dt, forecast, dist = batch['x'], batch['y'], batch['points'], batch['edge_index'], batch['dt'][0], batch['forecast'][0], batch['dist']
        y_hat = self.net(x, points, edge_index, dt, forecast, dist)
        loss = self.loss(y_hat, y)
        return loss, y_hat, y, x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss, y_hat, y, x = self.step(batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the valadation loop
        real = batch['real']
        w00 = self.net._modules['dt_block'].w00
        res0 = self.net._modules['dt_block'].linear.weight.data[0].flatten()
        res = torch.zeros(res0.size()[0]+1)
        for i in range(res0.size()[0]+1):
            if i==0: res[i] = w00
            else: res[i] = res0[i-1]
        if self.point_wise: res = torch.mean(res, 0)
        res = dict(zip(self.weight_dict, res.tolist()))
        self.log_dict(res)
        res = self.dict2tensor(res)
        real = self.dict2tensor(real)
        val_loss = self.loss(res.flatten(), real)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        points = batch['points']
        loss, y_hat, y, x = self.step(batch)
        w00 = self.net._modules['dt_block'].w00
        res0 = self.net._modules['dt_block'].linear.weight.data[0].flatten()
        res = torch.zeros(res0.size()[0]+1)
        for i in range(res0.size()[0]+1):
            if i==0: res[i] = w00
            else: res[i] = res0[i-1]
        if self.point_wise: res = torch.mean(res, 0)
        res = dict(zip(self.weight_dict, res.tolist()))
        self.printdict(res)
        if self.plot: plot.show_img_irregular(x=x, y_hat=y_hat, y=y, s='forecast_test_TPnet_'+self.equation, points=points[0])
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
    
    def dict2tensor(self, d):
        l = []
        for i in range(self.order+1):
            for j in range(self.order+1-i):
                s = f'u{i}{j}'
                try:
                    try:
                        l.append(d[s][0])
                    except:
                        l.append(d[s])
                except:
                    l.append(0.0)
        return torch.tensor(l, dtype=torch.float32)
    
    def printdict(self, dict):
        s = "\n".join("{0}: {1}".format(k, v)  for k,v in dict.items())
        print(s)
