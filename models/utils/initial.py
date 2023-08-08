
import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple

from scipy.interpolate import Rbf
from scipy.signal import convolve2d



def rbf_init(shape: Tuple[int], period: Tuple[int] = (4, 4), padding: int = 2, amplitude=0.5):
    x_low = np.linspace(0+0.5/period[0], 1-0.5/period[0], period[0], endpoint=True)
    y_low = np.linspace(0+0.5/period[1], 1-0.5/period[1], period[1], endpoint=True)
    x_low, y_low = np.meshgrid(x_low, y_low)

    x_high = np.linspace(0+0.5/shape[0], 1-0.5/shape[0], shape[0], endpoint=True)
    y_high = np.linspace(0+0.5/shape[1], 1-0.5/shape[1], shape[1], endpoint=True)
    x_high, y_high = np.meshgrid(x_high, y_high)

    z_low = np.random.randn(period[0], period[1])*amplitude

    # padding
    x_low = np.hstack([x_low-(i+1) for i in range(padding)] + [x_low] + [x_low+(i+1) for i in range(padding)])
    y_low = np.hstack([y_low for i in range(padding)] + [y_low] + [y_low for i in range(padding)])
    z_low = np.hstack([z_low for _ in range(padding)] + [z_low] + [z_low for _ in range(padding)])

    x_low = np.vstack([x_low for i in range(padding)] + [x_low] + [x_low for i in range(padding)])
    y_low = np.vstack([y_low-(i+1) for i in range(padding)] + [y_low] + [y_low+(i+1) for i in range(padding)])
    z_low = np.vstack([z_low for _ in range(padding)] + [z_low] + [z_low for _ in range(padding)])


    value_interp = Rbf(x_low, y_low, z_low, function="gaussian")
    z_high = value_interp(x_high,y_high)

    #smoothing filter
    a = 5
    smoothing_filter = np.ones((2*a+1, 2*a+1))
    smoothing_filter/= np.sum(smoothing_filter)
    z_high = convolve2d(z_high, smoothing_filter, mode='same', boundary='wrap')

    return z_high


def wave_init(shape: Tuple[int], amplitude=0.5):
    x_high = np.linspace(0+0.5/shape[0], 1-0.5/shape[0], shape[0], endpoint=True)
    y_high = np.linspace(0+0.5/shape[1], 1-0.5/shape[1], shape[1], endpoint=True)
    x_high, y_high = np.meshgrid(x_high, y_high)

    z_high = amplitude*np.sin(2*np.pi*x_high)*np.sin(2*np.pi*y_high)

    #smoothing filter
    a = 5
    smoothing_filter = np.ones((2*a+1, 2*a+1))
    smoothing_filter/= np.sum(smoothing_filter)
    z_high = convolve2d(z_high, smoothing_filter, mode='same', boundary='wrap')

    return z_high