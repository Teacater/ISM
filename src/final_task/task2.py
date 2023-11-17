import numpy as np

# Material Properties
E = 200  # Young's Modulus in units
nu = 0.27  # Poisson's ratio
sigma_T = 300
# Force
angle = -120

# Define Zone 1 and Zone 2
zone_1_line = ((-0.7071067811865475, 0), (0, -0.7071067811865475))
zone_2_line = ((4.619397662556434, 1.913417161825449), (1.9134171618254492, 4.619397662556434))


def calculate_B_matrix(node_coords):
    """Calculate the B matrix for a triangle element."""
    x1, y1 = node_coords[0]
    x2, y2 = node_coords[1]
    x3, y3 = node_coords[2]
    A = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1
    B = (1 / (2 * A)) * np.array([
        [b1, 0, b2, 0, b3, 0],
        [0, c1, 0, c2, 0, c3],
        [c1, b1, c2, b2, c3, b3]
    ])
    return B


def triangle_stiffness_matrix(node_coords, E, nu):
    """Calculate the stiffness matrix for a single triangle."""
    x1, y1 = node_coords[0]
    x2, y2 = node_coords[1]
    x3, y3 = node_coords[2]
    A = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))  # area
    D = (E / (1 - nu ** 2)) * np.array(
        [
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ]
    )
    B = calculate_B_matrix(node_coords)
    K = A * np.dot(np.dot(B.T, D), B)
    return K


def global_stiffness_matrix(nodes, triangles, E, nu):
    """Assemble the global stiffness matrix from individual triangle stiffness matrices."""
    dof = len(nodes) * 2  # Degrees of freedom
    K_global = np.zeros((dof, dof))
    for tri in triangles:
        node_indices = [int(i) for i in tri]
        node_coords = [nodes[i] for i in node_indices]
        K_tri = triangle_stiffness_matrix(node_coords, E, nu)
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    for l in range(2):
                        K_global[2 * node_indices[i] + k, 2 * node_indices[j] + l] += K_tri[2 * i + k, 2 * j + l]
    return K_global


def is_on_line_segment(point, line_start, line_end, tolerance=1e-6):
    """Check if a point is on a line segment within a certain tolerance."""
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    d_line = np.hypot(x2 - x1, y2 - y1)
    d1 = np.hypot(x - x1, y - y1)
    d2 = np.hypot(x - x2, y - y2)
    return abs(d1 + d2 - d_line) < tolerance


def apply_fixed_boundary_conditions(K_global, fixed_nodes):
    """Apply fixed boundary conditions to the global stiffness matrix."""
    for node in fixed_nodes:
        idx = node * 2  # 2 degrees of freedom
        K_global[idx:idx + 2, :] = 0
        K_global[:, idx:idx + 2] = 0
        K_global[idx, idx] = 1
        K_global[idx + 1, idx + 1] = 1
    return K_global


def calculate_nodal_forces_from_distributed_force(nodes, force_nodes, P, angle):
    """Calculate nodal forces from a distributed force acting at an angle."""
    F = np.zeros(len(nodes) * 2)
    angle_rad = np.radians(angle)
    fx = P * np.cos(angle_rad)
    fy = P * np.sin(angle_rad)
    for node in force_nodes:
        idx = node * 2
        F[idx] += fx
        F[idx + 1] += fy
    return F


def calculate_stress_strain_matrix(E, nu):
    """Calculate the plane stress D matrix for a given material."""
    D = (E / (1 - nu ** 2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])
    return D


def calculate_stress(nodes, triangles, displacements, E, nu):
    """Calculate stress for each triangle."""
    stresses = []
    for tri in triangles:
        node_indices = [int(i) for i in tri]
        node_coords = [nodes[i] for i in node_indices]
        displacement_vector = np.array([displacements[2 * i: 2 * i + 2] for i in node_indices]).flatten()
        B = calculate_B_matrix(node_coords)
        strain = np.dot(B, displacement_vector)
        D = calculate_stress_strain_matrix(E, nu)
        stress = np.dot(D, strain)
        stresses.append(stress)

    return stresses


def calculate_maximum_force(K_global_modified, nodes, force_nodes, triangles, E, nu, sigma_T, increment=0.1):
    """Calculate the maximum force P the detail can withstand without plastic deformation."""
    P_max = 0
    stress_max = 0
    while stress_max < sigma_T:
        P_max += increment
        F = calculate_nodal_forces_from_distributed_force(nodes, force_nodes, P_max, angle)
        displacements = np.linalg.solve(K_global_modified, F)
        stresses = calculate_stress(nodes, triangles, displacements, E, nu)
        stress_max = max(np.linalg.norm(stress) for stress in stresses)

    return P_max - increment  # Subtract the last increment as it caused the stress to exceed σ_T
