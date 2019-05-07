import numpy as np


class UnionFind(object):
    """
    Class thats implements find-union data structure. 
    Performs fast cluster labeling in unsorted dendrogram.

    Parameters
    ----------
    N: int
        Number of nodes in the graph.

    """
    def __init__(self, N):
        self.parent = np.arange(2*N - 1)
        self.size = np.ones(2*N - 1)
        self.nextlabel = N

    def union(self, m, n):
        """ merge two points into N + 1 cluster """
        self.parent[m] = self.nextlabel
        self.parent[n] = self.nextlabel

        new_size = self.size[m] + self.size[n]
        self.size[self.nextlabel] = new_size

        self.nextlabel += 1
        return new_size

    def find(self, n):
        """ finds out which cluster the point belongs to """
        p = n
        while self.parent[n] != n:
            n = self.parent[n]

        while self.parent[p] != n:
            p, self.parent[p] = self.parent[p], n

        return n
