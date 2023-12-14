import matplotlib.pyplot as plt
import numpy as np

def generate_points(delta_w, delta_r):
    points = []
    for W in np.arange(0, 80, delta_w):
        for R in np.arange(0, 84/2, delta_r):
            if W + 2*R < 80 and 2 * R < 84 and R > 0 and W > 0:
                points.append((W, R))
    return points


if __name__ == '__main__':
    points = generate_points(1, 1)
    W, R = zip(*points)
    plt.scatter(W, R, s=3)
    plt.xlabel('W')
    plt.ylabel('R')
    plt.title('Feasible Points for Given Constraints')
    plt.show()
