from matplotlib import pyplot as plt

from src.final_task2.task1 import compute_figure_4_data
import numpy as np
E = 200  # Young's Modulus in units
nu = 0.27  # Poisson's ratio
ro = 6800
# Force
angle = np.radians(-135)
force = 1000


def triangle_area(triangle):
    (x1, y1), (x2, y2), (x3, y3) = triangle
    return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))


def compute_figure_mass(nodes, triangles, ro):
    total_area = 0
    for triangle_indices in triangles:
        triangle_vertices = [nodes[i] for i in triangle_indices]
        total_area += triangle_area(triangle_vertices)
    width_m = 0.01
    volume_m3 = total_area * width_m
    mass_kg = volume_m3 * ro
    return mass_kg


def triangle_stiffness_matrix(triangle, E, nu):
    (x1, y1), (x2, y2), (x3, y3) = triangle
    A = triangle_area(triangle)

    B = np.array([
        [y2 - y3, 0, y3 - y1, 0, y1 - y2, 0],
        [0, x3 - x2, 0, x1 - x3, 0, x2 - x1],
        [x3 - x2, y2 - y3, x1 - x3, y3 - y1, x2 - x1, y1 - y2]
    ]) / (2 * A)

    D = (E / (1 - nu ** 2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])

    K = A * np.dot(np.dot(B.T, D), B)
    return K


def compute_figure_stiffness(nodes, triangles, E, nu):
    dof = 2 * len(nodes)
    global_stiffness = np.zeros((dof, dof))
    for triangle_indices in triangles:
        triangle_vertices = [nodes[i] for i in triangle_indices]
        K_local = triangle_stiffness_matrix(triangle_vertices, E, nu)
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    for l in range(2):
                        global_stiffness[2 * triangle_indices[i] + k, 2 * triangle_indices[j] + l] += K_local[
                            2 * i + k, 2 * j + l]

    return global_stiffness


def apply_force_and_boundary_conditions(nodes, triangles, force, angle, E, nu):
    dof = 2 * len(nodes)
    F = np.zeros(dof)  # Force vector
    fixed_nodes = [i for i, (x, y) in enumerate(nodes) if y == 0]
    force_nodes = [i for i, (x, y) in enumerate(nodes) if y == 1]
    force_per_node = force / len(force_nodes)
    force_x = force_per_node * np.cos(angle)
    force_y = force_per_node * np.sin(angle)
    for i in force_nodes:
        F[2*i] += force_x
        F[2*i + 1] += force_y

    K_global = compute_figure_stiffness(nodes, triangles, E, nu)

    # Apply boundary conditions (fix nodes at y = 0)
    for node in fixed_nodes:
        for dof in [2*node, 2*node+1]:
            K_global[dof, :] = 0
            K_global[:, dof] = 0
            K_global[dof, dof] = 1
            F[dof] = 0

    return K_global, F, fixed_nodes, force_nodes

def calculate_strain_energy(K_global, F):
    displacements = np.linalg.lstsq(K_global, F, rcond=None)[0]

    U = 0.5 * np.dot(displacements.T, np.dot(K_global, displacements))
    return U

def draw_figure(nodes, triangles, fixed_nodes, force_nodes, title):
    plt.figure(figsize=(10, 10))
    plt.triplot([node[0] for node in nodes], [node[1] for node in nodes], triangles, 'b-')
    plt.scatter([node[0] for node in nodes], [node[1] for node in nodes], color='red', s=5)
    plt.scatter([nodes[node][0] for node in fixed_nodes], [nodes[node][1] for node in fixed_nodes], color='blue', s=100)
    plt.scatter([nodes[node][0] for node in force_nodes], [nodes[node][1] for node in force_nodes], color='pink', s=100)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()


def calculate_toughness(nodes, triangles):
    K_global, F, fixed_nodes, forced_nodes = apply_force_and_boundary_conditions(nodes, triangles, force, angle, E, nu)
    # draw_figure(nodes, triangles, fixed_nodes, forced_nodes, "Figure 4")
    return calculate_strain_energy(K_global, F)


if __name__ == '__main__':
    nodes, triangles = compute_figure_4_data()
    mass = compute_figure_mass(nodes, triangles, ro)
    strain_energy = calculate_toughness(nodes, triangles)
    print(f"Strain Energy: {strain_energy} J")
    print(f"Mass of the figure: {mass} kg")
