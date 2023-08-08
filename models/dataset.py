import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Any
from .utils import knn
import re
from .utils import animation as a
from .utils import rand_sample

class pdeDataset(Dataset):
    """PDE dataset."""

    def __init__(self, npy_file,
                 root_directory,
                 train: bool,
                 DT: float = 0.1,
                 real: torch.Tensor=None,
                 forecast: int=1,
                 random_sample: bool=False,
                 k: int=5,
                 bound: float=32.0,
                 pad: bool=True,
                 transform=None,
                 animate: bool=False,
                 TP_order:int=2):
        """
        Arguments:
            npy_file (string): Path to the npy file + data|times.npy
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file = npy_file
        # Load Data
        if train:
            data = np.load(root_directory + "/" + npy_file + "_data.npy")
        else:
            data = np.load(root_directory + "/" + npy_file + "_test_data.npy")
        if animate:
            print("Making Animation")
            a.animate_solution(data, './data/'+self.file+'/'+self.file+'_animation.gif')
            print("Animation Created")
        points =  np.load(root_directory + "/" + npy_file + "_points.npy")
        # reconfigure points
        x = points[0]
        y = points[1]
        points = np.hstack([x.reshape(-1, 1), y.reshape(-1,1)])
        self.data = torch.from_numpy(data)
        ds = self.data.size()
        self.data = torch.reshape(self.data, (ds[0], ds[1] * ds[2]))
        self.points = torch.from_numpy(points)
        self.data = self.data.type(torch.float32)
        self.points = self.points.type(torch.float32)
        if random_sample>0: self.points, self.data = rand_sample.rand_sample_i(self.points, self.data, num_points=random_sample)
        self.dt = DT
        self.forecast = forecast
        self.real = real
        # KNN - Algorithm
        self.dist, self.edge_index = knn.knn_torch(points_from=self.points, points_to=self.points, k_neighbors=k, bound=bound, padding=pad)
        self.transform = transform
        # read equation file
        if real is None:
            f = open(root_directory + "/" + npy_file + "_equation.txt", 'r')
            str = f.readline()
            f.close()
            eq_params = self.parse_equation_from_string(str)
            #dict_param = dictionary for real values
            eq_value = [eq_params[i][0] for i in range(len(eq_params))]
            eq_names = [eq_params[i][1][0] for i in range(len(eq_params))]
            dict_param = dict(zip(eq_names, eq_value))
            self.set_real(dict_param)

    def __len__(self):
        return self.data.size()[0]-self.forecast

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {'x': self.data[idx],
                  'y': self.data[idx+1:idx+self.forecast+1],
                  'points': self.points,
                  'edge_index': self.edge_index,
                  'dt': self.dt,
                  'real': self.real,
                  'forecast': self.forecast,
                  'dist': self.dist}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def set_real(self, tensor_weights):
        self.real = tensor_weights
    
    def parse_equation_from_string(self, expression: str):
        terms = re.split("[+]", expression)
        def parse_term(term):
            parts = re.split("[*|^]", term)
            eq = [float(parts[0])]+[(parts[i], int(parts[i+1])) for i in range(1, len(parts), 2)]
            return eq
        return [parse_term(t) for t in terms]
