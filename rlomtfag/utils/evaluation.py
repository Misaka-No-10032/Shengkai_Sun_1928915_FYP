import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment


def best_map(l1, l2):
    """
    Permute labels of L2 to match L1 as good as possible.
    
    :param l1: numpy array
    :param l2: numpy array
    :return new_l2: numpy array
    """
    l1 = l1.ravel()
    l2 = l2.ravel()
    if len(l1) != len(l2):
        raise ValueError('size(l1) must == size(l2)')

    label1, inv1 = np.unique(l1, return_inverse=True)
    label2, inv2 = np.unique(l2, return_inverse=True)

    n_class1 = len(label1)
    n_class2 = len(label2)

    g_mat = np.zeros((n_class1, n_class2))
    np.add.at(g_mat, (inv1, inv2), 1)

    _, col_ind = linear_sum_assignment(-g_mat.T)

    new_l2 = label1[col_ind[inv2]]
    return new_l2


def mutual_info(x, y):
    # Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)).
    assert x.size == y.size, "Input arrays must have the same size."
    n = x.size
    x = x.flatten()
    y = y.flatten()

    min_val = min(np.min(x), np.min(y))
    x = x - min_val
    y = y - min_val
    max_val = max(np.max(x), np.max(y))

    # count the number of occurrences of each combination of values in x and y
    counts = np.bincount(x * (max_val + 1) + y, minlength=(max_val + 1) * (max_val + 1))
    p_xy = counts.reshape((max_val + 1, max_val + 1)).astype(float) / n
    p_xy = p_xy[p_xy > 0]
    h_xy = -np.dot(p_xy, np.log2(p_xy + np.finfo(float).eps).T)

    p_x = np.bincount(x, minlength=max_val + 1).astype(float) / n
    p_y = np.bincount(y, minlength=max_val + 1).astype(float) / n

    # entropy of p_y and p_x
    h_x = -np.dot(p_x, np.log2(p_x + np.finfo(float).eps).T)
    h_y = -np.dot(p_y, np.log2(p_y + np.finfo(float).eps).T)

    # mutual information
    mi = h_x + h_y - h_xy

    # normalized mutual information
    v = np.sqrt((mi / h_x) * (mi / h_y))
    return v.item()


def eval_results(h_mat, gnd, label=None):
    n_class = len(np.unique(gnd))
    if label is None:
        if isinstance(h_mat, list):
            h_mat = h_mat[-1]
        kmeans = KMeans(n_clusters=n_class, init='random', max_iter=100, n_init=1)
        kmeans.fit(h_mat.T)
        label = kmeans.labels_

    if gnd.shape != label.shape:
        label = label.reshape(-1, 1)

    mi_hat = mutual_info(gnd, label)

    res = best_map(gnd, label)
    acc = np.sum(gnd.ravel() == res) / len(gnd)

    return acc, mi_hat


if __name__ == '__main__':
    pass
