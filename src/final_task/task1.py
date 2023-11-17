import numpy as np
from scipy.spatial import Delaunay
from src.task1.task1 import split_polygon


def is_inside_polygon(point, polygon_vertices):
    """Check if a point is inside a polygon using ray-casting algorithm."""
    n = len(polygon_vertices)
    inside = False
    x, y = point
    for i in range(n):
        x1, y1 = polygon_vertices[i]
        x2, y2 = polygon_vertices[(i + 1) % n]
        if min(y1, y2) < y <= max(y1, y2) and x <= max(x1, x2):
            if y1 != y2:
                xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
            if x1 == x2 or x <= xinters:
                inside = not inside
    return inside


def compute_figure_e_data():
    octagon_coordinates = [
        (4.619397662556434, 1.913417161825449),
        (1.9134171618254492, 4.619397662556434),
        (-1.9134171618254485, 4.619397662556434),
        (-4.619397662556434, 1.9134171618254494),
        (-4.619397662556434, -1.9134171618254483),
        (-1.9134171618254516, -4.619397662556432),
        (1.91341716182545, -4.619397662556433),
        (4.619397662556432, -1.913417161825452)
    ]

    square_coordinates = [
        (0, 0.7071067811865475),
        (-0.7071067811865475, 0),
        (0, -0.7071067811865475),
        (0.7071067811865475, 0)
    ]

    segment_length = 0.3  # segment length

    octagon_nodes = split_polygon(octagon_coordinates, segment_length)
    square_nodes = split_polygon(square_coordinates, segment_length)

    x_coords, y_coords = zip(*octagon_coordinates)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Generate internal nodes
    internal_nodes = []
    for x in np.arange(x_min, x_max, segment_length):
        for y in np.arange(y_min, y_max, segment_length):
            point = (x, y)
            if is_inside_polygon(point, octagon_coordinates) and not is_inside_polygon(point, square_coordinates):
                internal_nodes.append(point)

    nodes = octagon_nodes + internal_nodes + square_nodes
    tri = Delaunay(nodes)

    # Filtering out triangles
    triangles = [triangle for triangle in tri.simplices if
                 not is_inside_polygon(np.mean(np.array(nodes)[triangle], axis=0), square_coordinates)]

    return nodes, triangles
