from math import *
from random import *
from argparse import ArgumentParser
import csv


def cluster(ps, clusters, iterations):
    """
        K-means clustering algorithm without numpy
        Parameters
        ----------
        ps: list of tuple
            list of input of point set
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
        >>> len(Counter(cluster(ps, 10, 10)))
        10
        """

    # Generate n centre points randomly, n is the number of clusters
    centres = []
    for i in range(clusters):
        centres.append(ps[randrange(len(ps))])

    alloc = [None] * len(ps)
    n = 0
    while n < iterations:
        for i in range(len(ps)):
            p = ps[i]
            # add a for loop to calculate the distance between each points to the centres
            d = [None] * clusters
            for j in range(clusters):
                d[j] = sqrt((p[0] - centres[j][0]) ** 2 + (p[1] - centres[j][1]) ** 2 + (p[2] - centres[j][2]) ** 2)
            # Find the shortest distance to the n centres and then return their index in point set.
            # The index of alloc means the index of all points, the corresponding number of index means
            # which cluster the point belong to.
            alloc[i] = d.index(min(d))
        for i in range(clusters):
            alloc_ps = [p for j, p in enumerate(ps) if alloc[j] == i]

            if len(alloc_ps) == 0:
                continue

            new_mean = (sum([a[0] for a in alloc_ps]) / len(alloc_ps), sum([a[1] for a in alloc_ps]) / len(alloc_ps),
                        sum([a[2] for a in alloc_ps]) / len(alloc_ps))
            centres[i] = new_mean
        n = n + 1

    return alloc


if __name__ == "__main__":
    parser = ArgumentParser(description="K-means clustering algorithm without numpy")
    parser.add_argument('filepath', help="Input file name", type=str)
    parser.add_argument("--iters", nargs="?", const="True", default=10, type=int)
    arguments = parser.parse_args()

    file = open(arguments.filepath, encoding='utf-8')
    csv_f = csv.reader(file)

    result_list = []
    for row in csv_f:
        result_list.append((float(row[0]), float(row[1]), float(row[2])))

    clusters = cluster(result_list, 3, arguments.iters)

    print("Result of clustering with iteration {}: {}".format(arguments.iters, clusters))
