import torch
import random

def gen_mask(grid_size: int, device : torch.device, dropout: float=0.05, seed : int = 42):
    torch.manual_seed(seed)
    x = torch.rand(grid_size * grid_size, device=device)
    mask = x.ge(dropout)
    return mask

def rand_sample(points : torch.Tensor, values : torch.Tensor, mask : torch.Tensor):
    '''
    points in Tensor    [ batch , grid_size x grid_size , 2]

    values in Tensor    [ batch , grid_size x grid_size]

    mask in Tensor      [ grid_size x grid_size ]
    '''
    batch = points.size()[0]
    mask_ = mask.unsqueeze(-1).expand(points.size())
    print(mask_)
    points = torch.masked_select(points, mask_)
    values = torch.masked_select(values, mask)
    points = torch.reshape(points, (batch, points.size()[0]//(batch*2), 2))
    values = torch.reshape(values, (batch, values.size()[0]//(batch)))
    return points, values

def rand_sample_i(points : torch.Tensor, values : torch.Tensor, num_points : int, seed : int = 42):
    '''
    points in Tensor    [ batch , grid_size x grid_size , 2]

    values in Tensor    [ batch , grid_size x grid_size]

    num_points          Number of points 

    (OPTIONAL) Seed     for random generator : 42
    '''
    random.seed(seed)
    i = random.sample(range(0, points.size()[0]), num_points)
    i = torch.tensor(i, dtype=torch.int32, device=points.device).flatten()
    i, index = torch.sort(i)
    points = torch.index_select(points, dim=0, index=i)
    values = torch.index_select(values, dim=1, index=i)
    return points, values

# test
if __name__ == '__main__':
    p = torch.tensor([[[0,0], [1,1], [0,1], [1,0]], [[0,0], [1,1], [0,1], [1,0]]], device='cuda')
    v = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], device='cuda')
    #mask = gen_mask(2, p.device, 0.5)
    #p, v = rand_sample(p, v, mask)
    p, v = rand_sample_i(p, v, 2, 42)

    print(p)
    print(v)