import numpy as np
from argparse import ArgumentParser
import csv


def cluster_numpy(ps, clusters=3, iterations=10):
    """
    K-means clustering algorithm with numpy
    Parameters
    ----------
    ps: tuple
        input of point set
    clusters: int
        the required number of clusters
    iterations: int
        the number of iteration
    Returns
    -------
    alloc: list
        a lists with the track indices that belong to each group
    Examples
    --------
    >>> from collections import Counter
    >>> lines = open('samples.csv', 'r').readlines()
    >>> ps=[]
    >>> for line in lines: ps.append(tuple(map(float, line.strip().split(','))))
    >>> len(Counter(cluster_numpy(ps, 10, 10)))
    10
    """
    ds = np.array(ps)

    # m is the number of samples and n is the number of features
    m, n = ds.shape
    alloc = np.empty(m, dtype=int)
    # choose clusters randomly
    centres = ds[np.random.choice(np.arange(m), clusters, replace=False)]

    iteration = 0
    while iteration < iterations:
        d = np.square(np.repeat(ds, clusters, axis=0).reshape(m, clusters, n) - centres)

        # ndarray(m, k), the distance between each sample and k centers of mass, total m rows
        distance = np.sqrt(np.sum(d, axis=2))

        # Index number of the nearest center of mass for each sample
        index_min = np.argmin(distance, axis=1)

        # If the sample clustering is not changed, then return allo
        if (index_min == alloc).all():
            # return alloc, centres
            return alloc.tolist()

        # reclassified and traverse the center of mass set
        alloc[:] = index_min
        for i in range(clusters):
            items = ds[alloc == i]
            centres[i] = np.mean(items, axis=0)
        iteration += 1


if __name__ == "__main__":
    parser = ArgumentParser(description="K-means clustering algorithm with numpy")
    parser.add_argument('filepath', help="Input file name", type=str)
    parser.add_argument("--iters", nargs="?", const="True", default=10, type=int)
    arguments = parser.parse_args()

    file = open(arguments.filepath, encoding='utf-8')
    csv_f = csv.reader(file)

    result_list = []
    for row in csv_f:
        result_list.append((float(row[0]), float(row[1]), float(row[2])))

    clusters = cluster_numpy(result_list, 3, arguments.iters)

    print("Result of cluster_numpy with iteration {}: {}".format(arguments.iters, clusters))