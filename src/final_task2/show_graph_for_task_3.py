import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import minimize

from src.final_task2.show_graph_for_task_1 import generate_points
from src.final_task2.task1 import compute_figure_4_data
from src.final_task2.task2 import calculate_toughness, ro, compute_figure_mass

P1_coords = None
P2_coords = None
J_values = None
K_values = None


if __name__ == '__main__':
    w_r_pairs = generate_points(10, 10)
    mass_values = []
    strength_values = []
    x, y = zip(*w_r_pairs)
    P1, P2 = list(x), list(y)
    for w, r in w_r_pairs:
        nodes, triangles = compute_figure_4_data(w, r)
        mass = compute_figure_mass(nodes, triangles, ro)
        strength = calculate_toughness(nodes, triangles)
        mass_values.append(mass)
        strength_values.append(strength)
    P1_coords = x
    P2_coords = y
    J_values = strength_values
    K_values = strength_values
    alphas = np.linspace(0, 1, 10)

    def interpolate_values(P1, P2, values):
        return griddata((P1_coords, P2_coords), values, (P1, P2), method='linear')

    def F(P, alpha):
        P1, P2 = P
        J = interpolate_values(P1, P2, J_values)
        K = interpolate_values(P1, P2, K_values)
        return alpha * J + (1 - alpha) * K

    def optimize_function(alpha, initial_guess):
        result = minimize(lambda P: -F(P, alpha), x0=initial_guess,
                          bounds=[(min(P1_coords), max(P1_coords)), (min(P2_coords), max(P2_coords))])
        return result.x, -result.fun
    optimal_solutions = [optimize_function(alpha, initial_guess=[0.5, 0.5]) for alpha in alphas]

    P1_values = [sol[0][0] for sol in optimal_solutions]
    P2_values = [sol[0][1] for sol in optimal_solutions]
    J_values_optimal = [interpolate_values(p1, p2, J_values) for p1, p2 in zip(P1_values, P2_values)]
    K_values_optimal = [interpolate_values(p1, p2, K_values) for p1, p2 in zip(P1_values, P2_values)]
    F_values = [sol[1] for sol in optimal_solutions]
    plt.figure()
    plt.scatter(J_values_optimal, K_values_optimal)
    plt.xlabel('σ (J)')
    plt.ylabel('m (K)')
    plt.title('Criteria Plane')
    plt.show()
    plt.figure()
    plt.scatter(P1_values, P2_values)
    plt.xlabel('P1')
    plt.ylabel('P2')
    plt.title('Parameter Plane')
    plt.show()
    table = pd.DataFrame(
        {'P1': P1_values, 'P2': P2_values, 'σ (J)': J_values_optimal, 'm (K)': K_values_optimal, 'F': F_values})
    print(table)

