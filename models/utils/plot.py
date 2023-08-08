import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tempfile import TemporaryFile
import torch
import numpy as np
import os

def plot(u: torch.Tensor, derivatives: torch.Tensor, N:int, du: torch.Tensor, result: torch.Tensor):
    derivatives = torch.transpose(derivatives, 0, 1)
    derivatives = torch.reshape(derivatives, (derivatives.size()[0], N, N))
    derivatives = derivatives.cpu()
    fig, ax = plt.subplots(1,8)
    ax[0].imshow(derivatives[0]) #dy
    ax[1].imshow(derivatives[1]) #dyy
    ax[2].imshow(derivatives[2]) #dx
    ax[3].imshow(derivatives[3]) #dxy
    ax[4].imshow(derivatives[4]) #dxx
    ax[5].imshow(u.cpu())
    ax[6].imshow(du.detach().cpu()) #dxx
    ax[7].imshow(result.detach().cpu()) #dxx
    ax[0].set_title('dy')
    ax[1].set_title('dyy')
    ax[2].set_title('dx')
    ax[3].set_title('dxy')
    ax[4].set_title('dxx')
    ax[5].set_title('u')
    ax[6].set_title('du')
    ax[7].set_title('result')
    plt.show()
    #fig.savefig('results/derivative')
    plt.close()


def show_img(x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor, s: str, N: int):
    fig, ax = plt.subplots(1,4)
    fig.tight_layout(pad=0.1)
    # set the spacing between subplots
    x_plt = torch.reshape(x[0], (N, N)).cpu().detach()
    y_plt = torch.reshape(y[0,-1], (N, N)).cpu().detach()
    r_plt = torch.reshape(y_hat[0, -1], (N, N)).cpu().detach()
    e_plt = r_plt - y_plt
    im0 = ax[0].imshow(x_plt.numpy())
    im1 = ax[1].imshow(y_plt.numpy())
    im2 = ax[2].imshow(r_plt.numpy())
    im3 = ax[3].imshow(e_plt.numpy())
    ax[0].set_title('Before [x]')
    ax[1].set_title('After [y]')
    ax[2].set_title('Computed [y_hat]')
    ax[3].set_title('Error-Map')
    divider = make_axes_locatable(ax[3])
    #cax = divider.append_axes("right", size="5%", pad=0.08)
    #fig.colorbar(im3, cax)
    data = np.stack([x_plt.numpy(), y_plt.numpy(), r_plt.numpy(), e_plt.numpy()], axis=0)
    fname = 'results/'+s+'.npy'
    fname = uniquify(fname)
    np.save(fname, data)
    #fig.savefig('results/' + s)
    #plt.show()
    #plt.close()

def show_img_irregular(x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor, s: str, points: torch.Tensor):
    fig, ax = plt.subplots(1,4)
    fig.tight_layout(pad=0.1)
    points_xy = points.permute(1, 0)
    dx = points_xy[0].cpu().detach().numpy().flatten()
    dy = points_xy[1].cpu().detach().numpy().flatten()
    # set the spacing between subplots
    y_plt = y[0].cpu().detach().numpy()
    r_plt = y_hat[0].cpu().detach().numpy()
    e_plt = r_plt - y_plt
    data = np.stack([y_plt, r_plt, e_plt], axis=0)
    data_points = np.stack([dx, dy], axis=0)
    fname = 'results/'+s+'.npy'
    fname = uniquify(fname)
    np.save(fname, data)
    np.save(fname, data)
    np.save(fname+'_points', data_points)

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1
    filename, extension = os.path.splitext(path)
    return filename