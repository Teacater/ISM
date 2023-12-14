import numpy as np
from scipy.spatial import Delaunay
from src.task1.task2 import split_arc
from src.task1.task3 import draw_figure


def generate_internal_nodes_for_square(polygon_vertices, segment_length):
    x_coords, y_coords = zip(*polygon_vertices)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    internal_nodes = []
    for x in np.arange(x_min, x_max + segment_length, segment_length):
        for y in np.arange(y_min, y_max + segment_length, segment_length):
            internal_nodes.append((x, y))
    return internal_nodes


def _on_line(p, line):
    (x1, y1), (x2, y2) = line
    if x1 == x2:  # Vertical line
        return p[0] == x1 and y1 <= p[1] <= y2
    elif y1 == y2:  # Horizontal line
        return p[1] == y1 and x1 <= p[0] <= x2
    else:  # Non-axis aligned line
        return (p[0] - x1) * (y2 - y1) == (p[1] - y1) * (x2 - x1)


def remove_nodes_from_lines(nodes, lines):
    return [node for node in nodes if not any([_on_line(node, line) for line in lines])]


def max_edge_length_of_triangle(triangle):
    edge_lengths = [np.linalg.norm(triangle[i] - triangle[(i + 1) % 3]) for i in range(3)]
    return max(edge_lengths)

def delaunay_triangulation_with_max_edge_filter(points, max_edge_length_threshold):
    tri = Delaunay(points)
    filtered_triangles = []
    for simplex in tri.simplices:
        triangle = [points[index] for index in simplex]
        if max_edge_length_of_triangle(np.array(triangle)) <= max_edge_length_threshold:
            filtered_triangles.append(simplex)

    return filtered_triangles


def compute_figure_4_data(w=10, r=30, segment_length=2):
    # W + 2R <= 80, W >= 0, R >= 0, 2 * R >= 84
    polygons = [
        [(0, 0), (0, 8), (80, 8), (80, 0)],
        [(40 - w / 2, 92), (40 + w / 2, 92), (40 + w / 2, 8), (40 - w / 2, 8)],
        [(0, 92), (0, 100), (80, 100), (80, 92)]
    ]
    remove_edges = [
        [(40 - w / 2 - r, 8), (40 - w / 2 - r, 8)],
        [(40 - w / 2 - r, 92), (40 - w / 2 - r, 92)],
        [(40 - w / 2, 8), (40 - w / 2, 8 + r)],
        [(40 + w / 2, 8), (40 + w / 2, 8 + r)],
        [(40 - w / 2, 92), (40 - w / 2, 92 - r)],
        [(40 + w / 2, 92), (40 + w / 2, 92 - r)],
    ]
    arcs = [
        ((40 - w / 2 - r, 8 + r), 3 * np.pi / 2, 2 * np.pi),
        ((40 + w / 2 + r, 8 + r), np.pi, 3 * np.pi / 2),
        ((40 - w / 2 - r, 92 - r), 0, np.pi / 2),
        ((40 + w / 2 + r, 92 - r), np.pi / 2, np.pi),
    ]
    small_polygons = [
        ((40 - w / 2 - r, 8 + r), (40 - w / 2, 8 + r), (40 - w / 2, 8), (40 - w / 2 - r, 8)),
        ((40 + w / 2 + r, 8 + r), (40 + w / 2 + r, 8), (40 + w / 2, 8), (40 + w / 2, 8 + r)),
        ((40 - w / 2 - r, 92 - r), (40 - w / 2 - r, 92), (40 - w / 2, 92), (40 - w / 2, 92 - r)),
        ((40 + w / 2 + r, 92 - r), (40 + w / 2, 92 - r), (40 + w / 2, 92), (40 + w / 2 + r, 92)),
    ]
    arc_nodes = []

    for polygon_vertices, ((x, y), _, _) in zip(small_polygons, arcs):
        arc_nodes += [i for i in generate_internal_nodes_for_square(polygon_vertices, segment_length) if (i[0] - x) ** 2 + (i[1] - y) ** 2 > r * r]
    internal_nodes = []
    for polygon_vertices in polygons:
        internal_nodes += generate_internal_nodes_for_square(polygon_vertices, segment_length)
    internal_nodes = remove_nodes_from_lines(internal_nodes, remove_edges)
    for arc in arcs:
        arc_nodes += split_arc(*arc, segment_length, r)
    nodes = internal_nodes + arc_nodes
    nodes = [(x / 100, y / 100) for (x, y) in nodes]

    nodes = list(set(nodes))
    return nodes, delaunay_triangulation_with_max_edge_filter(nodes, segment_length * 2 / 100)


if __name__ == '__main__':
    nodes, triangles = compute_figure_4_data()
    draw_figure(nodes, triangles, "Figure 4")

