from pde import (
    ScalarField, 
    CartesianGrid, 
    FieldCollection, 
    PDEBase, 
    FieldBase, 
    UnitGrid, 
    CallbackTracker,
    ProgressTracker
)

from pde.trackers.base import TrackerCollection

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from scipy.interpolate import Rbf


from eq_generator import EqGenerator
from initial import rbf_init
from animation import animate_solution


class PolynomialPDE(PDEBase):
    def __init__(self, equation, bc = 'auto_periodic_neumann') -> None:
        super().__init__()

        self.equation = equation
        self.bc = bc

    def get_partial_derivative(self, u, derivative_str):
        bc = 'auto_periodic_neumann'
        derivative_components = derivative_str.split('_')
        order_x = int(derivative_components[2])
        order_y = int(derivative_components[4])
        for i in range(order_x):
            u = u.gradient(bc)[0]
        for i in range(order_y):
            u = u.gradient(bc)[1]
        return u
    
    def extract_coordinates_from_grid(self, grid):
        shape = grid.shape
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        x = (grid.cell_coords[:,:,0]-x_bounds[0])/(x_bounds[1]-x_bounds[0])
        y = (grid.cell_coords[:,:,1]-y_bounds[0])/(y_bounds[1]-y_bounds[0])
        return x, y



    def get_term(self, u, term):
        result = term[0]
        for t in term[1:]:
            result *= self.get_partial_derivative(u, t[0])**t[1]
        return result
    
    def get_initial_state(self, grid):
        """prepare a useful initial state"""
        x, y = self.extract_coordinates_from_grid(grid)

        # initialize fields
        u = ScalarField(grid, rbf_init((64, 64)), label="u")
        #u = ScalarField(grid, self.rbf_init(x, y, period=10), label="u")
        return u, x, y
    
    def evolution_rate(self, state: FieldBase, t: float = 0) -> FieldBase:
        terms = [self.get_term(state, term) for term in self.equation]
        derivative = sum(terms)

        return sum(terms)
    


def solve_equation(equation: str, save_interval=0.01, tmax = 1):

    eq = EqGenerator.parse_equation_from_string(equation)
    print(eq)

    p = PolynomialPDE(eq)

    grid = CartesianGrid([[0,8],[0,8]],[64, 64], periodic=True)

    state, x, y = p.get_initial_state(grid)


    # setup saving equation states
    data = []
    times = []
    def save_state(state, time):
        data.append(state.copy().data)
        times.append(time)


    tracker_callback = CallbackTracker(save_state, interval=save_interval)
    tracker_progress = ProgressTracker(interval=save_interval)
    tracker = TrackerCollection([tracker_callback, tracker_progress])


    # solve
    sol = p.solve(state, t_range=(0, tmax), tracker=tracker)
    data = np.stack(data)
    times = np.stack(times)
    flat_points = np.hstack([x.reshape(-1, 1), y.reshape(-1,1)])
    return data, times, flat_points


if __name__ == "__main__":

    np.random.seed(42)

    eq = "1.097*u_x_1_y_0^1+2.945*u_x_0_y_1^1+0.219*u_x_2_y_0^1+0.32*u_x_0_y_2^1+0.786*u_x_1_y_1^1"
    # eq = "0.912*u_x_1_y_0^1+1.123*u_x_1_y_0^1+0.421*u_x_2_y_0^1+0.012*u_x_0_y_2^1"
    # eq = "0.234*u_x_1_y_0^1+0.234*u_x_1_y_0^1+1.234*u_x_2_y_0^1+2.096*u_x_0_y_2^1"

    # run solver
    data, times, flatpoints = solve_equation(eq)

    # save numpy arrays
    np.save("data/eq_data.npy", data)
    np.save("data/eq_times.npy", times)
    np.save("data/eq_points.npy", flatpoints)
    
    # to read arrays use
    # data = np.load("eq_data.npy")
    # times = np.load("eq_times.npy")

    # make a gif of the simulation
    animate_solution(data)

    # visualize start and end state
    fig, ax = plt.subplots(1,2)

    ax[0].imshow(data[0], label=f"u at t = 0", vmin=-0.5, vmax=0.5)
    ax[1].imshow(data[-1], label=f"u at t = {times[-1]}", vmin=-0.5, vmax=0.5)

    plt.show()


