import numpy as np

from .utils import nearest_min, update_distmatrix
from .utils import sort_linkage, label


def linkage_naive(dist_matrix):
    """
    Performs single linkage clustering on pairwise dinstance matrix with naive algorithm. 
    Takes O(N^3) time and O(N^2) memory.

    Parameters
    ----------
    dist_matrix: ndarray, shape (n, n)
        Matrix of pairwise distances.

    Returns
    -------
    Z: ndarray, shape (n - 1, 4)
        Sorted stepwise dendogram (linkage matrix in scipy standart).
    """
    dist_matrix = dist_matrix.copy()  # ? not the best choice
    linkage_matrix = _linkage_naive_core(dist_matrix)
    # sort dendogram by dists
    linkage_matrix = sort_linkage(linkage_matrix)
    # compute correct clusters labels and sizes
    linkage_matrix = label(linkage_matrix)

    return linkage_matrix


def _linkage_naive_core(dist_matrix):
    """
       Helper function. Performs naive single linkage and 
       returns unsorted dendogram with clusters representatives.

    Parameters
    ----------
    dist_matrix: ndarray, shape (n, n)
        Matrix of pairwise distances.

    Returns
    -------
    Z: ndarray
        Unsorted stepwise dendogram.
    """
    linkage_matrix = np.zeros((dist_matrix.shape[0] - 1, 4))
                  
    for k in range(dist_matrix.shape[0] - 1):
        i, j = nearest_min(dist_matrix)
        linkage_matrix[k, :3] = [i, j, dist_matrix[i, j]]
        # updates inplace!
        dist_matrix = update_distmatrix((i, j), dist_matrix)
          
    return linkage_matrix


def mst_linkage(labels, dist_matrix):
    """
    Performs single linkage clustering on pairwise dinstance matrix with MST algorithm. 
    Takes O(N^2) time and O(N) memory.

    Parameters
    ----------
    dist_matrix: ndarray, shape (n, n)
        Matrix of pairwise distances.

    Returns
    -------
    Z: ndarray, shape (n - 1, 4)
        Sorted stepwise dendogram (linkage matrix in scipy standart).
    """
    linkage_matrix = _mst_linkage_core(labels, dist_matrix)
    linkage_matrix = sort_linkage(linkage_matrix)
    linkage_matrix = label(linkage_matrix)

    return linkage_matrix


def _mst_linkage_core(labels, dist_matrix):
    """
       Helper function. 
       Implements MST (Minimum Spanning Tree) algorithm for single linkage.
       Based on Prim's algorithm.

    Parameters
    ----------
    dist_matrix: ndarray, shape (n, n)
        Matrix of pairwise distances.

    Returns
    -------
    Z: ndarray
        Unsorted stepwise dendogram.
    """
    N = dist_matrix.shape[0]
    linkage_matrix = np.zeros((N - 1, 4))

    dists = np.full(N, fill_value=np.inf)

    #  tracking the nodes that have already been passed
    merged = np.zeros_like(labels)
    
    # choose start node
    c = 0
    for i in range(N - 1):
        merged[c] = 1
        for node in labels:
            if merged[node] == 1:
                continue
            dists[node] = np.minimum(dists[node], dist_matrix[node, c])
        n = np.argmin(dists)

        linkage_matrix[i] = [c, n, dists[n], 0]
        dists[n] = np.inf
        c = n
    return linkage_matrix
