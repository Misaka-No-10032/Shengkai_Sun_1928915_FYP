import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
from rlomtfag.rlomtfag import rlomtfag
from rlomtfag.utils.mat_utils import nndsvd
from sklearn.metrics import pairwise_distances


class RLOMTFAGCluster():
    def __init__(self):
        self.param = None
        self.limiter = None
        self.epsilon = None
        self.distance_metric = None
        self.X = None
        self.k = None
        self.kmeans = None
        self.cluster_centers_ = None
        self._n_features_out = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.W = None
        self.S = None
        self.H = None

    def _prepare_data(self, X):
        if X.min() < 0:
            X -= X.min()
        r_x = np.sum(X, axis=1)
        # find indices of rows with zero elements
        zero_row_indices = np.where(r_x == 0)[0]
        # delete rows with zero elements
        X = np.delete(X, zero_row_indices, axis=0)

        # Calculate pairwise distances
        if self.distance_metric == 'sqeuclidean':
            xv = X / 255
        elif self.distance_metric == 'cosine':
            xv = X / np.max(X)
        else:
            raise TypeError(f'Unknown distance matrix: {self.distance_metric}')
        d_dv = pairwise_distances(xv.T, xv.T, metric=self.distance_metric)
        return X, d_dv

    def fit(self, payload):
        self.param = payload['param']
        self.limiter = payload['limiter']
        self.epsilon = payload['epsilon']
        self.distance_metric = payload['distance_metric']
        self.X = payload['X']
        self.k = payload['k']
        X, d_dv = self._prepare_data(self.X)
        u_tv, v_v = nndsvd(X, int(self.param[4]), 0)
        stv, vv = nndsvd(v_v, self.k, 0)
        vv = vv.T
        w_mat, s_mat, h_mat, gv_mat, obj, result = rlomtfag(X, u_tv, stv, vv, d_dv, self.param[0], self.param[1], self.param[2], self.limiter, self.epsilon, int(self.param[3]), self.distance_metric)
        # Cluster the data (hv_mat)
        self.kmeans = KMeans(n_clusters=self.k, max_iter=100, n_init=10)
        self.kmeans.fit(h_mat)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self._n_features_out = self.cluster_centers_.shape[1]
        self.labels_ = self.kmeans.labels_
        self.inertia_ = self.kmeans.inertia_
        self.n_iter_ = self.kmeans.n_iter_
        self.W = w_mat
        self.S = s_mat
        self.H = h_mat
        return self

    def project(self, X_new):
        # Project X onto the learned subspace
        if X_new.min() < 0:
            X_new -= X_new.min()
        return np.linalg.pinv(self.W @ self.S) @ X_new



if __name__ == '__main__':
    pass
