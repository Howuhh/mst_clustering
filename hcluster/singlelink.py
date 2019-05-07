import numpy as np

from scipy.cluster.hierarchy import dendrogram, fcluster

from .utils import pairwise_dist
from .hierarchy import linkage_naive, mst_linkage


class SingleLinkClust(object):
    """
    Single linkage agglomerative clustering.
    
    Merges the pair of clusters that minimally increases
    a euclidean linkage distance on each iteration.

    Parameters
    ----------
    n_clusters: int
        Number of clusters.

    method:  {"naive", "mst"}
        Which linkage algorithm to use.

    """
    def __init__(self, n_clusters, method="naive"):
        self.n_clusters = n_clusters
        self.method = method
        
        #TODO: add labels_ propery

        if method not in ("naive", "mst"):
            raise ValueError("Invalid method, must be naive or mst")
        
        if self.n_clusters < 0:
            raise ValueError("Number of clusters must be positive")


    def fit(self, df):
        """ Compute linkage matrix and returns model """
        if not isinstance(df, np.ndarray):
            raise ValueError("Data must be ndarray type")

        dist_matrix = pairwise_dist(df)
        labels = np.arange(df.shape[0])

        if self.method == "naive":
            self.linkage_matrix = linkage_naive(dist_matrix)
        else:
            self.linkage_matrix = mst_linkage(labels, dist_matrix)

        return self

    def fit_predict(self, df):
        """ Fit model and returns clusters labels """
        _ = self.fit(df)
        self.labels = fcluster(self.linkage_matrix, self.n_clusters, criterion='maxclust')

        return self.labels
