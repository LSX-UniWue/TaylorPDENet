# compute the derivative for each vertex
# INPUT: vertex + list of neighbors - 2, 5, ... ~ according to Taylor Polynomial
# OUTPUT: list of derivatives

# need to solve a system of equations

import torch
import math as m

# for one point
def solveDerivative(vertex: torch.Tensor=torch.zeros(1),
                     neighbors: torch.Tensor=torch.zeros(2),
                     vertex_value: torch.Tensor=torch.zeros(1),
                     neighbors_value: torch.Tensor=torch.zeros(2),
                     order: int=1) -> torch.Tensor:
    """
    Input: a vertex and its neighbors and order for Taylor-Polynomial

    Return: tensor of derivatives at vertex

    Attention: order of TP and Number of neighbors needs to be coherent
    """
    # check
    sum = 0
    for x in range(order):
        sum = sum+x+1
    if(neighbors.size()[0] != sum+order):
        raise RuntimeError("Order and number of neighbors do not match")
    A_size = (neighbors.size()[0], sum+order)
    # compute distances
    distances = torch.sub(neighbors, vertex)
    #print(distances)
    # compute matrix for Taylorpolynomial
    # each row is now only x or y
    distances_sorted = torch.transpose(distances, 0, 1)
    #print(distances_sorted)
    # generate A
    A = torch.tensor([])
    for i in range(order+1):
        for j in range(order+1-i):
            if (i==0 and j==0): continue
            c = torch.pow(distances_sorted[0], i) * torch.pow(distances_sorted[1], j) / (m.factorial(i)*m.factorial(j))
            c = torch.reshape(c, (1, neighbors.size()[0]))
            A = torch.cat((A, c), 0)
    rand = torch.rand(A_size) * 0.000000001
    A = A + rand
    A = torch.transpose(A, 0, 1)
    A = torch.reshape(A, A_size)
    b = torch.sub(neighbors_value, vertex_value)
    #print(f"A = {A}")
    #print(f"b = {b}")
    # Solve with torch linalg System Ax = b
    x = torch.linalg.solve(A, b)
    #print(f"x = {x}")
    # y - y² - x - xy - x² -> for order 2
    # y - y² - y³ - x - xy - xy² - x²y - x³ -> for order 3
    return x

# for severeal points
def solveDerivatives(vertices: torch.Tensor,
                     neighbors: torch.Tensor,
                     vertices_values: torch.Tensor,
                     neighbors_values: torch.Tensor,
                     num_derivatives: int,
                     knn: int,
                     order: int,
                     dist: torch.Tensor) -> torch.Tensor:
    """
    Input: a vertex and its neighbors and order for Taylor-Polynomial

    Ensure_solve adds random numbers * epsilon to the x and y distances

    Return: tensor of derivatives at vertex

    Attention: order of TP and Number of neighbors needs to be coherent
    """
    # compute distances - uncomment if without padding
    # vertices_s = torch.repeat_interleave(vertices, knn, dim=0)
    # vertices_s = torch.reshape(vertices_s, (vertices.size()[0]*knn, vertices.size()[-1]))
    # vertices_s = torch.transpose(vertices_s, 0, 1)
    # neighbors_s = torch.reshape(neighbors, (neighbors.size()[0]*knn, neighbors.size()[-1]))
    # neighbors_s = torch.transpose(neighbors_s, 0, 1)
    # # neighbors and vertices have shape [2, Size*TP]
    # distances_y = torch.sub(neighbors_s[0], vertices_s[0])
    # distances_x = torch.sub(neighbors_s[1], vertices_s[1])

    distances_y = dist[:,:,0].flatten()
    distances_x = dist[:,:,1].flatten()
    
    # compute matrix for Taylorpolynomial
    # each row is now only x or y
    # generate A
    A = torch.tensor([], device=neighbors.device)
    for i in range(order+1):
        for j in range(order+1-i):
            if i==0 and j ==0: continue
            c = torch.pow(distances_x, i) * torch.pow(distances_y, j) / (m.factorial(i)*m.factorial(j))
            c = torch.reshape(c, (1, neighbors.size()[0]*neighbors.size()[1]))
            A = torch.cat((A, c), 0)
    A = torch.transpose(A, 0, 1)
    A = torch.reshape(A, (vertices.size()[0], neighbors.size()[1], num_derivatives))
    # compute right hand side of equation: b
    vertices_values_ = torch.repeat_interleave(vertices_values, knn, dim=0)
    vertices_values_ = torch.reshape(vertices_values_, (vertices_values.size()[0], knn))
    b = torch.sub(neighbors_values, vertices_values_)
    b = torch.reshape(b, (vertices.size()[0], neighbors.size()[1], 1))
    # Solve with torch linalg System Ax = b
    derivatives = torch.linalg.lstsq(A, b)
    # y yy x xy xx
    return derivatives[0]