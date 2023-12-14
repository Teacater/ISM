import numpy as np
from ipywidgets import interact, fixed, FloatSlider
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

from src.final_task2.show_graph_for_task_1 import generate_points
from src.final_task2.task1 import compute_figure_4_data
from src.final_task2.task2 import compute_figure_mass, ro, calculate_toughness


def create_contour_plot(x, y, result_values, title):
    Z = np.array(result_values)
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), Z, (Xi, Yi), method='linear')
    plt.contourf(Xi, Yi, Zi, levels=14, cmap="RdBu_r")
    plt.colorbar()
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.title(title)
    plt.show()

def create_combined_contour_plot(x, y, J_values, K_values, alpha, title):
    Z_J = np.array(J_values)
    Z_K = np.array(K_values)
    combined_Z = alpha * Z_J + (1 - alpha) * Z_K

    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), combined_Z, (Xi, Yi), method='linear')

    plt.contourf(Xi, Yi, Zi, levels=14, cmap="RdBu_r")
    plt.colorbar()
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.title(f"{title} (alpha={alpha:.2f})")
    plt.show()



if __name__ == '__main__':
    w_r_pairs = generate_points(10, 10)
    mass_values = []
    strength_values = []
    x, y = zip(*w_r_pairs)
    x, y = list(x), list(y)
    for w, r in w_r_pairs:
        nodes, triangles = compute_figure_4_data(w, r)
        mass = compute_figure_mass(nodes, triangles, ro)
        strength = calculate_toughness(nodes, triangles)
        mass_values.append(mass)
        strength_values.append(strength)
    create_contour_plot(x, y, mass_values, "Contour plot of mass")
    create_contour_plot(x, y, strength_values, "Contour plot of mass")
    plt.figure()
    plt.scatter(mass_values, strength_values)
    plt.title("Scatter Plot on Criteria Plane")
    plt.xlabel("Mass")
    plt.ylabel("Strength Criterion")
    plt.show()
    interact(create_combined_contour_plot,
             x=fixed(x),
             y=fixed(y),
             J_values=fixed(mass_values),
             K_values=fixed(strength_values),
             alpha=FloatSlider(min=0, max=1, step=0.01, value=0.5),
             title=fixed('Combined Contour Plot'))
    plt.show()

