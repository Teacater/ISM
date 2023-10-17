import numpy as np
from matplotlib import pyplot as plt

from src.task1.task3 import compute_figure_1_data, compute_figure_2_data

E = 210e9  # Young's Modulus (Pa)
v = 0.3  # Poisson's Ratio

# Elasticity matrix for plane stress
D = (E / (1 - v ** 2)) * np.array([
    [1, v, 0],
    [v, 1, 0],
    [0, 0, (1 - v) / 2]
])


def compute_element_stiffness(triangle, nodes):
    A, B = calc_B_array(nodes, triangle)
    return A * np.dot(B.T, np.dot(D, B))


def assemble_global_stiffness(nodes, triangles):
    n_dofs = 2 * len(nodes)  # 2 DOFs (u and v) for each node
    K = np.zeros((n_dofs, n_dofs))
    for triangle in triangles:
        k_e = compute_element_stiffness(triangle, nodes)
        for i in range(3):
            for j in range(3):
                K[2 * triangle[i]:2 * triangle[i] + 2, 2 * triangle[j]:2 * triangle[j] + 2] \
                    += k_e[2 * i:2 * i + 2, 2 * j:2 * j + 2]

    return K


def calc_B_array(nodes, triangle):
    x, y = zip(*[nodes[i] for i in triangle])

    A = 0.5 * (x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1]))

    b = [y[1] - y[2], y[2] - y[0], y[0] - y[1]]
    c = [x[2] - x[1], x[0] - x[2], x[1] - x[0]]

    return A, np.array([
        [b[0], 0, b[1], 0, b[2], 0],
        [0, c[0], 0, c[1], 0, c[2]],
        [c[0], b[0], c[1], b[1], c[2], b[2]]
    ]) / (2 * A)


def compute_element_strain(triangle, nodes, displacements):
    A, B = calc_B_array(nodes, triangle)

    d_element = np.array([displacements[2 * triangle[0]], displacements[2 * triangle[0] + 1],
                          displacements[2 * triangle[1]], displacements[2 * triangle[1] + 1],
                          displacements[2 * triangle[2]], displacements[2 * triangle[2] + 1]])

    strain = np.dot(B, d_element)
    return strain


def compute_element_stress(strain):
    return np.dot(D, strain)


def apply_boundary_conditions(K, nodes):
    # fix nodes on the left side.
    fixed_nodes = [i for i, (x, _) in enumerate(nodes) if abs(x - np.min(np.array(nodes)[:, 0])) < 1e-5]

    for node in fixed_nodes:
        K[2 * node, :] = 0
        K[:, 2 * node] = 0
        K[2 * node, 2 * node] = 1

        K[2 * node + 1, :] = 0
        K[:, 2 * node + 1] = 0
        K[2 * node + 1, 2 * node + 1] = 1

    return K


def compute_principal_stresses():
    nodes, triangles = compute_figure_2_data()
    K = assemble_global_stiffness(nodes, triangles)
    F = np.zeros(2 * len(nodes))
    F[-2] = 1000  # force for Y axis
    # F[-1] = 100  # force for X axis
    apply_boundary_conditions(K, nodes)

    displacements = np.linalg.lstsq(K, F, rcond=None)[0]
    strains, stresses = [], []
    for triangle in triangles:
        strain = compute_element_strain(triangle, nodes, displacements)
        stress = compute_element_stress(strain)

        strains.append(strain)
        stresses.append(stress)

    principal_stresses = []
    for stress in stresses:
        sigma_xx = stress[0]
        sigma_yy = stress[1]
        tau_xy = stress[2]

        sigma1 = 0.5 * (sigma_xx + sigma_yy) + np.sqrt(0.25 * (sigma_xx - sigma_yy) ** 2 + tau_xy ** 2)
        sigma2 = 0.5 * (sigma_xx + sigma_yy) - np.sqrt(0.25 * (sigma_xx - sigma_yy) ** 2 + tau_xy ** 2)

        principal_stresses.append((sigma1, sigma2))

    return nodes, triangles, principal_stresses


def visualize_principal_stress(nodes, triangles, principal_stresses):
    nodes = np.array(nodes)
    triangles = np.array(triangles)

    stress_magnitudes = [max(stress) for stress in principal_stresses]

    plt.figure(figsize=(10, 7))
    plt.tripcolor(nodes[:, 0], nodes[:, 1], triangles, facecolors=stress_magnitudes, edgecolors='k', cmap='jet')
    plt.colorbar(label='Principal Stress (Pa)')
    plt.title('Principal Stress Distribution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    visualize_principal_stress(*compute_principal_stresses())
