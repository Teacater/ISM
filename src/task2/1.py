import numpy as np
from matplotlib import pyplot as plt

from src.task1.task3 import compute_figure_1_data, compute_figure_2_data


def barycentric_coordinates(p, a, b, c):
    detT = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    alpha = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / detT
    beta = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / detT
    gamma = 1.0 - alpha - beta
    return alpha, beta, gamma


def is_point_in_triangle(p, a, b, c):
    alpha, beta, gamma = barycentric_coordinates(p, a, b, c)
    return 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1


def interp_2d(nodes, elements, phi, x, y):
    point = (x, y)
    for triangle in elements:
        a, b, c = nodes[triangle[0]], nodes[triangle[1]], nodes[triangle[2]]
        if is_point_in_triangle(point, a, b, c):
            alpha, beta, gamma = barycentric_coordinates(point, a, b, c)
            return alpha * phi[triangle[0]] + beta * phi[triangle[1]] + gamma * phi[triangle[2]]
    return None


def compute_triangle_average_phi(triangle, phi_values):
    return sum([phi_values[i] for i in triangle]) / 3


def main():
    nodes_1, triangles_1 = compute_figure_1_data()
    phi_values = [np.exp(-node[0] /1000) for node in nodes_1]  # Example function

    triangle_phi_values = [compute_triangle_average_phi(triangle, phi_values) for triangle in triangles_1]

    phi_max = max(triangle_phi_values)
    phi_min = min(triangle_phi_values)

    plt.figure(figsize=(10, 10))
    plt.tripcolor([node[0] for node in nodes_1], [node[1] for node in nodes_1], triangles_1,
                  facecolors=triangle_phi_values, edgecolors='k', vmin=phi_min, vmax=phi_max)
    plt.colorbar(label='Phi Value')

    test_points = [(np.random.randint(0, 120), np.random.randint(0, 110)) for _ in range(10)]
    offset = 20  # distance outside the main figure to display the color points

    for point in test_points:
        interpolated_value = interp_2d(nodes_1, triangles_1, phi_values, *point)
        if interpolated_value is not None:
            outside_point = (point[0] - offset, point[1])
            plt.scatter(*outside_point, c=[interpolated_value], edgecolors='black', s=100, vmin=phi_min, vmax=phi_max)
            plt.plot([point[0], outside_point[0]], [point[1], outside_point[1]], 'k-')

    plt.title('Delaunay Triangulation with Interpolated Phi Values')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
