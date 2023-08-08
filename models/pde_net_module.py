from typing import Any
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn.functional import mse_loss
from .PDE_NET2.polypde import POLYPDE2D
from .utils import plot

class PDENet(nn.Module):
    def __init__(self,
                 input_dim,
                 max_order,
                 kernel_size=5,
                 constraint='frozen', 
                 hidden_layers=2, 
                 scheme='upwind',
                 dt=1,
                 dx=1):
        super().__init__()
        self.num_channel = input_dim
        channel_names_str = 'u'
        self.pdenet = POLYPDE2D(
            dt=dt,
            dx=dx,
            kernel_size=kernel_size,
            max_order=max_order,
            constraint=constraint,
            channel_names=channel_names_str,
            hidden_layers=hidden_layers,
            scheme=scheme
        ).to(torch.float32)

    def forward(self, x, num_steps):
        x = torch.unsqueeze(x, dim=1) # introduces channel dim
        return self.pdenet.multistep(x, num_steps)

class PDENetModule(LightningModule):
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
        self.plot = plot
        self.order = order
        self.N = N
        self.point_wise = point_wise
        self.equation = equation

    def step(self, batch: Any, test: bool=False):
        x, y, forecast = batch['x'], batch['y'], batch['forecast'][0]
        x = torch.reshape(x, (x.size()[0], self.N, self.N))
        if test:
            with torch.inference_mode(False):
                y_hat = self.net(x, forecast)
        else:
            y_hat = self.net(x, forecast)
        y_hat = torch.reshape(y_hat, (x.size()[0], forecast, self.N*self.N))
        loss = self.loss(y_hat, y)
        return loss, y_hat, y, x

    def training_step(self, batch: Any, batch_idx: int):
        loss, y_hat, y, x = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        # compare weights with real pde
        real = batch['real']
        dict_coeffs = self.coeffs()
        try:
            self.log_dict(dict_coeffs)
        except:
            pass
        real = self.dict2tensor(real)
        coeffs = self.dict2tensor(dict_coeffs)
        val_loss = self.loss(coeffs, real)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch: Any, batch_idx: int):
        points = batch['points']
        loss, y_hat, y, x = self.step(batch, test=True)
        dict_coeffs = self.coeffs()
        self.printdict(dict_coeffs)
        if self.plot: plot.show_img_irregular(x=x, y_hat=y_hat, y=y, s='forecast_test_PDEnet_'+self.equation, points=points[0])
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.NAdam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

    def coeffs(self):
        model = self.net.pdenet
        dict_coeffs = None # dictionary for values
        for poly in model.polys:
            try:
                names, values = poly.coeffs()
                names = [f'{i}' for i in names]
                dict_coeffs = dict(zip(names, values))
            except:
                pass
        return dict_coeffs
    
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