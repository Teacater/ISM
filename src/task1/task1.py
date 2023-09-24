import numpy as np
import matplotlib.pyplot as plt


def split_polygon(vertices, segment_length):
    result_points = []
    for i in range(len(vertices)):
        start_point = np.array(vertices[i])
        end_point = np.array(vertices[(i + 1) % len(vertices)])
        distance = np.linalg.norm(end_point - start_point)
        num_segments = int(np.ceil(distance / segment_length))

        for j in range(num_segments):
            point = start_point + (end_point - start_point) * (j / num_segments)
            result_points.append(point.tolist())

    return result_points


def example_split_polygon():
    polygon = [[0, 0], [0, 10], [10, 10], [10, 0]]
    segment_length = 5
    points = split_polygon(polygon, segment_length)
    x, y = zip(*points)
    x = list(x) + [x[0]]
    y = list(y) + [y[0]]

    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, '-o', markersize=6, label='Split Points')
    orig_x, orig_y = zip(*polygon)
    orig_x = list(orig_x) + [orig_x[0]]
    orig_y = list(orig_y) + [orig_y[0]]
    plt.plot(orig_x, orig_y, '--', label='Original Polygon')
    plt.legend()
    plt.grid(True)
    plt.title('Polygon Split Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == "__main__":
    example_split_polygon()
