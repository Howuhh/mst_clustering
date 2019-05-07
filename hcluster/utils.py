import numpy as np

from hcluster.union import UnionFind


def label(linkage_matrix):
    """ function that finds correct clust labels from unsorted dendogram """
    N = linkage_matrix.shape[0] + 1
    Union = UnionFind(N)

    for i in range(N - 1):
        a, b = np.int(linkage_matrix[i, 0]), np.int(linkage_matrix[i, 1])
        a_new, b_new = Union.find(a), Union.find(b)

        # link-by-index:  linking the root node with
        # lower index to the root node with higher index
        if a_new < b_new:
            linkage_matrix[i, :2] = [a_new, b_new]
        else:
            linkage_matrix[i, :2] = [b_new, a_new]

        linkage_matrix[i, 3] = Union.union(a_new, b_new)

    return linkage_matrix


def sort_linkage(linkage_matrix):
    """ sort linkage matrix by distance between clusters """
    order = np.argsort(linkage_matrix[:, 2], kind='mergesort')
    linkage_matrix = linkage_matrix[order]
    
    return linkage_matrix


def pairwise_dist(X):
    """ computes paiwise distances """
    dist_matrix = (X**2).sum(axis=1, keepdims=True) + (X**2).sum(axis=1) - 2*np.dot(X, X.T)
    _ = np.fill_diagonal(dist_matrix, np.inf)

    return np.sqrt(dist_matrix)


def nearest_min(dist_matrix):
    """ finds minimum indices in an array"""
    # much faster than np.where
    i, j = np.unravel_index(
        np.argmin(dist_matrix), 
        dims=dist_matrix.shape
    )
    return i, j


def update_distmatrix(min_idx, dist_matrix):
    """ 
    Updates distance matrix inplace, sets megred clusters to inf 
    and computes the distances from new cluster to others.
    """
    i, j = min_idx
    new_cluster_dist = np.minimum(dist_matrix[i, :], dist_matrix[j, :])
    new_cluster_dist[i] = np.inf
    
    dist_matrix[i, :] = new_cluster_dist
    dist_matrix[:, i] = new_cluster_dist
    
    dist_matrix[j, :] = np.inf
    dist_matrix[:, j] = np.inf
    
    return dist_matrix