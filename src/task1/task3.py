import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from src.task1.task1 import split_polygon
from src.task1.task2 import split_arc


def is_inside_circle(point, center, radius):
    """Check if a point is inside a circle."""
    return np.linalg.norm(np.array(point) - np.array(center)) < radius


def is_inside_hexagon(point, center, radius):
    hexagon_angles = np.linspace(0, 2*np.pi, 7)[:-1]
    hexagon_vertices = [(center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)) for angle in hexagon_angles]
    sum_angles = 0
    for i in range(6):
        angle1 = np.arctan2(hexagon_vertices[i][1] - point[1], hexagon_vertices[i][0] - point[0])
        angle2 = np.arctan2(hexagon_vertices[(i+1)%6][1] - point[1], hexagon_vertices[(i+1)%6][0] - point[0])
        dtheta = angle2 - angle1
        while dtheta > np.pi:
            dtheta -= 2 * np.pi
        while dtheta < -np.pi:
            dtheta += 2 * np.pi
        sum_angles += dtheta
    return np.isclose(sum_angles, 2*np.pi)


def compute_figure_1_data():
    polygon_vertices = [(25, 90), (0, 90), (25, 105), (115, 105), (115, 0), (95, 0), (95, 20)]
    arc_center = (25, 20)
    arc_radius = 70
    segment_length = 5
    arc_step_length = 5

    raw_polygon_nodes = split_polygon(polygon_vertices, segment_length)
    arc_nodes = split_arc(arc_center, 0, np.pi / 2, arc_step_length, arc_radius)

    polygon_nodes = [node for node in raw_polygon_nodes if not is_inside_circle(node, arc_center, arc_radius)]

    x_min, y_min = np.min(np.array(polygon_vertices), axis=0)
    x_max, y_max = np.max(np.array(polygon_vertices), axis=0)
    internal_nodes = [(x, y) for x in np.arange(x_min + 5, x_max, 10) for y in np.arange(y_min + 5, y_max, 10) if
                      not is_inside_circle((x, y), arc_center, arc_radius)]

    nodes = polygon_nodes + arc_nodes + internal_nodes
    tri = Delaunay(nodes)
    triangles = [triangle for triangle in tri.simplices if
                 not is_inside_circle(np.mean(np.array(nodes)[triangle], axis=0), arc_center, arc_radius)]

    return nodes, triangles


def compute_figure_2_data():
    segment_length = 10
    arc_step_length = 5
    main_frame_vertices = np.array([(0, 0), (0, 100), (210, 100), (210, 0)])
    hexagon_center = (60, 45)
    circle_center = (165, 45)
    hexagon_radius = 15
    circle_radius = 20

    main_frame_nodes = split_polygon(main_frame_vertices, segment_length)
    hexagon_angles = np.linspace(0, 2*np.pi, 7)[:-1]
    hexagon_vertices = [(hexagon_center[0] + hexagon_radius * np.cos(angle), hexagon_center[1] + hexagon_radius * np.sin(angle)) for angle in hexagon_angles]
    hexagon_nodes = split_polygon(hexagon_vertices, segment_length)
    circle_nodes = split_arc(circle_center, 0, 2*np.pi, arc_step_length, circle_radius)

    internal_nodes = [(x, y) for x in np.arange(5, 205, 10) for y in np.arange(5, 95, 10) if not (is_inside_circle((x, y), circle_center, circle_radius) or is_inside_hexagon((x, y), hexagon_center, hexagon_radius))]
    nodes = main_frame_nodes + hexagon_nodes + circle_nodes + internal_nodes
    tri = Delaunay(nodes)
    triangles = [triangle for triangle in tri.simplices if not (is_inside_circle(np.mean(np.array(nodes)[triangle], axis=0), circle_center, circle_radius) or is_inside_hexagon(np.mean(np.array(nodes)[triangle], axis=0), hexagon_center, hexagon_radius))]

    return nodes, triangles


def draw_figure(nodes, triangles, title):
    plt.figure(figsize=(10, 10))
    plt.triplot([node[0] for node in nodes], [node[1] for node in nodes], triangles, 'b-')
    plt.scatter([node[0] for node in nodes], [node[1] for node in nodes], color='red', s=5)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    nodes_1, triangles_1 = compute_figure_1_data()
    draw_figure(nodes_1, triangles_1, 'Delaunay Triangulation of Part A')

    nodes_2, triangles_2 = compute_figure_2_data()
    draw_figure(nodes_2, triangles_2, 'Delaunay Triangulation of Part B')
