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
    def __init__(self, method="naive"):
        self.method = method
        self.linkage_matrix = None
        
        # TODO: add labels_ propery

        if method not in ("naive", "mst"):
            raise ValueError("Invalid method, must be naive or mst")

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

    def predict(self, n_clusters):
        if self.linkage_matrix is None:
            raise ValueError("Linkage matrix does not exits. First, fit the model")
                
        if n_clusters < 0:
            raise ValueError("Number of clusters must be positive")

        self.labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        
        return self.labels

    def fit_predict(self, df, n_clusters):
        """ Fit model and returns clusters labels """
        _ = self.fit(df)
        _ = self.predict(n_clusters)

        return self.labels
