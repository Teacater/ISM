import numpy as np
import matplotlib.pyplot as plt


def split_arc(center, start_angle, end_angle, arc_step_length, radius):
    arc_length = radius * (end_angle - start_angle)
    num_points = int(np.ceil(arc_length / arc_step_length))
    angles = np.linspace(start_angle, end_angle, num_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return list(zip(x, y))


def example_split_arc():
    center = [0, 0]
    start_angle = 0
    end_angle = np.pi / 2
    arc_step_length = 0.5
    radius = 5
    points = split_arc(center, start_angle, end_angle, arc_step_length, radius)
    x, y = zip(*points)
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, '-o', markersize=6, label='Arc Points')
    plt.scatter(*center, color='red', label='Center')
    plt.legend()
    plt.grid(True)
    plt.title('Arc Points Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([-radius - 1, radius + 1])
    plt.ylim([-radius - 1, radius + 1])
    plt.show()


if __name__ == "__main__":
    example_split_arc()
