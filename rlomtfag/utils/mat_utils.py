import numpy as np
from scipy.linalg import svd
from scipy.spatial.distance import cdist


def l2_distance_1(a, b):
    # if a and b are row vectors, turn them into matrix R^(2, size(a,2))
    if a.ndim == 1:
        a = a.reshape([1, -1])
        b = b.reshape([1, -1])

    d = cdist(a.T, b.T, 'sqeuclidean')
    return d


def eu_dist_2(fea_a, fea_b=None, b_sqrt=True):
    if fea_b is None:
        fea_b = fea_a

    d = cdist(fea_a, fea_b, 'euclidean')
    if not b_sqrt:
        d = np.square(d)
    return d


def e_proj_simplex(v, k=1):
    """
    The projection algorithm proposed by Duchi et al. (2008) 
    to project a vector onto the positive simplex with sum k. 
    This implementation has a worst-case time complexity of O(nlogn):
    NOTE: This algorithm is different from the original algorithm used in RLOMTFAG code (in MATLAB).
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - k
    ind = np.arange(n) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def nndsvd(a_mat, k, flag=0):
    # check the input matrix
    if np.any(a_mat < 0):
        raise ValueError('The input matrix contains negative elements !')

    # size of the input matrix
    m, n = a_mat.shape
    # the matrices of the factorization
    w_mat = np.zeros((m, k))
    h_mat = np.zeros((k, n))

    # 1st SVD --> partial SVD rank-k to the input matrix a_mat.
    u_mat, s_mat, v_mat = svd(a_mat)
    u_mat = u_mat[:, :k]
    s_mat = s_mat[:k]
    v_mat = v_mat[:k, :].T
    # choose the first singular triplet to be non-negative
    w_mat[:, 0] = np.sqrt(s_mat[0]) * np.abs(u_mat[:, 0])
    h_mat[0, :] = np.sqrt(s_mat[0]) * np.abs(v_mat[:, 0])

    # 2nd SVD for the other factors
    for i in range(1, k):
        uu = u_mat[:, i]
        vv = v_mat[:, i]
        uup = np.maximum(uu, 0)
        uun = np.maximum(-uu, 0)
        vvp = np.maximum(vv, 0)
        vvn = np.maximum(-vv, 0)
        n_uup = np.linalg.norm(uup)
        n_vvp = np.linalg.norm(vvp)
        n_uun = np.linalg.norm(uun)
        n_vvn = np.linalg.norm(vvn)
        term_p = n_uup * n_vvp
        term_n = n_uun * n_vvn
        if term_p >= term_n:
            w_mat[:, i] = np.sqrt(s_mat[i]*term_p)*uup/n_uup
            h_mat[i, :] = np.sqrt(s_mat[i]*term_p)*vvp/n_vvp
        else:
            w_mat[:, i] = np.sqrt(s_mat[i]*term_n)*uun/n_uun
            h_mat[i, :] = np.sqrt(s_mat[i]*term_n)*vvn/n_vvn

    # Actually these numbers are zeros
    w_mat[w_mat < 1e-10] = 0.1
    h_mat[h_mat < 1e-10] = 0.1

    # NNDSVDa: fill in the zero elements with the average
    if flag == 1:
        ind1 = np.where(w_mat == 0)
        ind2 = np.where(h_mat == 0)
        average = np.mean(a_mat[:])
        w_mat[ind1] = average
        h_mat[ind2] = average

    # NNDSVDar: fill in the zero elements with random values in the space [0:average/100]
    elif flag == 2:
        ind1 = np.where(w_mat == 0)
        ind2 = np.where(h_mat == 0)
        n1 = np.size(ind1)
        n2 = np.size(ind2)
        average = np.mean(a_mat[:])
        w_mat[ind1] = (average * np.random.rand(n1) / 100)
        h_mat[ind2] = (average * np.random.rand(n2) / 100)

    return w_mat, h_mat


if __name__ == '__main__':
    pass
