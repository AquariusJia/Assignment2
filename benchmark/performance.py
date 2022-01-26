import time
import tracknaliser.clustering_numpy
import tracknaliser.clustering
from matplotlib import pyplot as plt
import numpy as np


def performance():
    """
    Comparing the running time between cluster function without numpy
    and cluster function with numpy.
    """
    # creating increasing number of point set, ranging from 100 to 1000
    ps = np.random.rand(1000, 3)
    number_of_points = np.linspace(100, 1000, 91)
    run_time = []
    run_time_np = []
    for i in range(99, 1000, 10):
        # Get running time for cluster function without numpy for each increasing point set
        start = time.time()
        tracknaliser.clustering.cluster(ps[:i, :], clusters=3, iterations=10)
        end = time.time()
        run_time.append(end - start)

        # Get running time of cluster function with numpy for each increasing point set
        start_np = time.time()
        tracknaliser.clustering_numpy.cluster_numpy(ps[:i, :], clusters=3, iterations=10)
        end_np = time.time()
        run_time_np.append(end_np - start_np)

    plt.plot(number_of_points, run_time, label='cluster_without_numpy')
    plt.plot(number_of_points, run_time_np, label='cluster_with_numpy')
    plt.xlabel('number of points')
    plt.ylabel('time')
    plt.legend()
    plt.title('Comparing running time with two versions of cluster function')
    plt.savefig('performance.png')


if __name__ == "__main__":
    performance()
