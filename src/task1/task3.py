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


def draw_figure_1():
    polygon_vertices = [(25, 90), (0, 90), (25, 105), (115, 105), (115, 0), (95, 0), (95, 20)]
    arc_center = (25, 20)
    arc_radius = 70
    segment_length = 5
    arc_step_length = 5

    polygon_points = split_polygon(polygon_vertices, segment_length)
    arc_points = split_arc(arc_center, 0, np.pi / 2, arc_step_length, arc_radius)
    internal_nodes = polygon_points + arc_points
    tri = Delaunay(internal_nodes)
    valid_triangles = [triangle for triangle in tri.simplices if
                       not any(is_inside_circle(internal_nodes[vertex], arc_center, arc_radius) for vertex in triangle)]
    valid_nodes = [node for node in internal_nodes if not is_inside_circle(node, arc_center, arc_radius)]
    plt.figure(figsize=(10, 10))
    plt.triplot([x[0] for x in internal_nodes], [x[1] for x in internal_nodes], valid_triangles, 'b-')
    plt.scatter([x[0] for x in valid_nodes], [x[1] for x in valid_nodes], color='red', s=5)
    plt.title('Delaunay Triangulation of Part A')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()


def draw_figure_2():
    segment_length = 10
    arc_step_length = 5
    main_frame_vertices = np.array([(0, 0), (0, 100), (210, 100), (210, 0)])
    hexagon_center = (60, 45)
    circle_center = (165, 45)
    hexagon_radius = 15
    circle_radius = 20

    main_frame_points = split_polygon(main_frame_vertices, segment_length)
    hexagon_angles = np.linspace(0, 2*np.pi, 7)[:-1]
    hexagon_vertices = [(hexagon_center[0] + hexagon_radius * np.cos(angle), hexagon_center[1] + hexagon_radius * np.sin(angle)) for angle in hexagon_angles]
    hexagon_points = split_polygon(hexagon_vertices, segment_length)
    circle_points = split_arc(circle_center, 0, 2*np.pi, arc_step_length, circle_radius)

    internal_points = [(x, y) for x in np.arange(5, 205, 10) for y in np.arange(5, 95, 10) if not (is_inside_circle((x, y), circle_center, circle_radius) or is_inside_hexagon((x, y), hexagon_center, hexagon_radius))]
    all_points = np.vstack([main_frame_points, hexagon_points, circle_points, internal_points])
    tri = Delaunay(all_points)

    valid_simplices = [simplex for simplex in tri.simplices if not (is_inside_circle(np.mean(all_points[simplex], axis=0), circle_center, circle_radius) or is_inside_hexagon(np.mean(all_points[simplex], axis=0), hexagon_center, hexagon_radius))]

    plt.figure(figsize=(10, 10))
    plt.triplot(all_points[:, 0], all_points[:, 1], valid_simplices, 'b-')
    plt.scatter(all_points[:, 0], all_points[:, 1], color='red', s=5)
    plt.title('Delaunay Triangulation of Part B')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    draw_figure_1()
    draw_figure_2()
