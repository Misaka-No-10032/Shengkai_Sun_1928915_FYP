import time
import numpy as np
from scipy.io import loadmat
from rlomtfag.rlomtfag import rlomtfag
from rlomtfag.utils.mat_utils import nndsvd
import matplotlib.pyplot as plt
from rlomtfag.utils.evaluation import eval_results
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from rlomtfag.utils.data_gen import two_moon_gen, three_ring_gen
from sklearn.metrics.pairwise import euclidean_distances


def adaptive_learning_ability():
    new_data = 1
    datatype = 2  # 1: two-moon data, 2: three-ring data,

    if new_data == 1:
        if datatype == 1:
            num0 = 100
            x = two_moon_gen(num0)
            c = 2
            y = np.concatenate([np.ones(num0), 2 * np.ones(num0)])
        elif datatype == 2:
            num0 = 500
            x, y, _, _, _ = three_ring_gen(num0, 0.05)
            c = 3

    x1 = x[y == 1]
    x2 = x[y == 2]
    if datatype == 2:
        x3 = x[y == 3]
    x = x.T
    xv = x / 255
    d_dv = euclidean_distances(xv.T)

    nn, mm = x.shape
    r = c
    k = 5

    point_size = 75

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.scatter(x1[:, 0], x1[:, 1], s=point_size, c='r', label='data point 1')
    ax.scatter(x2[:, 0], x2[:, 1], s=point_size, c='b', label='data point 2')
    if datatype == 2:
        ax.scatter(x3[:, 0], x3[:, 1], s=point_size, c='g', label='data point 3')
    ax.legend(loc='upper right', fontsize=11, handlelength=1, markerscale=1)

    # options = {'WeightMode': 'Binary', 'k': k}
    # g_mat = construct_w(x.T, options)

    cv = c * c
    vv = np.random.rand(mm, r)
    utv = np.random.rand(nn, cv)
    stv = np.random.rand(cv, r)
    limiter = 0
    lambda_val = -8
    alphav = -2
    beta = -6
    pv = k + 5
    _, _, _, gv_mat, _, _ = rlomtfag(
        x, utv, stv, vv, d_dv, 10 ** lambda_val, 10 ** alphav, 10 ** beta, limiter, 10 ** -10, pv)

    s2_mat = gv_mat
    name = 'Eq. (2)'
    kk = pv

    ns2, ms2 = np.where(s2_mat.toarray() > 0)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    for i in range(len(ns2)):
        f21 = [x[1, ns2[i]], x[1, ms2[i]]]
        f22 = [x[0, ns2[i]], x[0, ms2[i]]]
        h1 = plt.plot(f22, f21, '-k', linewidth=1)

    p1 = plt.scatter(x1[:, 0], x1[:, 1], s=point_size, c='r')
    p2 = plt.scatter(x2[:, 0], x2[:, 1], s=point_size, c='b')
    if datatype == 2:
        p3 = plt.scatter(x3[:, 0], x3[:, 1], s=point_size, c='g')
        ax.legend([p1, p2, p3, h1[0]], ['data point 1', 'data point 2', 'data point 3', f"{name} with k={kk}"],
                  loc='upper right')
    else:
        ax.legend([p1, p2, h1[0]], ['data point 1', 'data point 2', f"{name} with k={kk}"], loc='upper right')

    plt.show()


def cluster_ability():
    # Define the dataset names and their parameters
    names = ['ORL_32x32', 'Yale_32x32', 'COIL100', 'UMIST',
             'JAFFE', 'BBCSport', 'UCI_Digits', 'Reuters-21578', 'oral']
    parameters = np.array([[-10, -4, -4, 9, 46],
                           [-5, 3, -6, 9, 23],
                           [-6, 4, -7, 14, 160],
                           [-9, -3, -2, 7, 27],
                           [-10, -4, -4, 8, 18],
                           [-4, -2, -5, 4, 5],
                           [-7, -2, -3, 10, 12],
                           [-7, -4, -5, 3, 4]])
    
    for iii in [3]:  # Select the dataset indices to run the code for
        # Load the dataset
        selection = names[iii]
        data = loadmat('./data/' + selection + '.mat')
        # Convert uint8 to int64 to avoid overflow
        data['fea'] = data['fea'].astype(np.float64)
        data['gnd'] = data['gnd'].astype(int)
        x = data['fea']
        min_val = x.min()
        if min_val < 0:
            x -= min_val

        # Prepare the data
        total = np.unique(data['gnd'])
        exclude = np.array([])
        ceshi = np.setdiff1d(total, exclude)
        nr = len(ceshi)
        rr = np.array([])
        t_label = np.array([])
        for i in range(nr):
            lr = np.where(data['gnd'] == ceshi[i])[1].reshape((1, -1))
            if lr.shape[0] > lr.shape[1]:
                rr = np.vstack([rr, lr]) if rr.size > 0 else lr
            else:
                rr = np.vstack([rr, lr.T]) if rr.size > 0 else lr.T
            t_label = np.vstack([t_label, np.ones((lr.shape[1], 1)) * (i + 1)]) if t_label.size > 0 else np.ones(
                (lr.shape[1], 1)) * (i + 1)
        # Keep the data in x where the columns index in rr
        x = np.hstack([x[:, r] for r in rr])

        r_x = np.sum(x, axis=1)
        # find indices of rows with zero elements
        zero_row_indices = np.where(r_x == 0)[0]
        # delete rows with zero elements
        x = np.delete(x, zero_row_indices, axis=0)

        # Calculate pairwise distances and set distance metric based on the dataset
        if selection in ['ORL_32x32', 'Yale_32x32', 'COIL100', 'UMIST', 'JAFFE', 'oral']:
            xv = x / 255
            distance_metric = 'sqeuclidean'
        else:
            xv = x / np.max(x)
            distance_metric = 'cosine'
        d_dv = pairwise_distances(xv.T, xv.T, metric=distance_metric)

        limiter = 100
        epsilon = 1e-10
        times = 10
        avg = np.zeros((times, 2))
        time_arr = np.zeros((times, 1))

        for t in range(times):
            u_tv, v_v = nndsvd(x, parameters[iii, 4], 0)
            stv, vv = nndsvd(v_v, nr, 0)
            vv = vv.T
            t0 = time.time()
            w_mat, s_mat, hv_mat, _, _, _ = rlomtfag(x, u_tv, stv, vv, d_dv, 10 ** int(parameters[iii, 0]),
                                             10 ** int(parameters[iii, 1]), 10 ** int(
                    parameters[iii, 2]), limiter, epsilon, int(parameters[iii, 3]),
                                             'sqeuclidean' if distance_metric == 'sqeuclidean' else 'cosine')
            t1 = time.time()
            time_arr[t] = t1 - t0
            accuracy_avg, mi_hat_avg = eval_results(hv_mat.T, t_label.astype(int))
            avg[t, :] = [accuracy_avg, mi_hat_avg]

        # ddof=1 for sample standard deviation
        result = np.vstack((np.mean(avg, axis=0), np.std(avg, axis=0, ddof=1)))
        mean_time = np.mean(time_arr)
        print(mean_time)
        print(result)
        return mean_time, result


def collaborative_ability():
    n1 = 600
    n2 = 300
    n3 = 0
    n = 50

    rand1 = np.random.rand(4) * n + n1
    rand2 = np.random.rand(4) * n + n2
    rand3 = np.random.rand(4) * n + n3

    rand13 = np.concatenate([rand1, rand3], axis=0)
    rand23 = np.concatenate([rand2, rand3], axis=0)
    rand31 = np.concatenate([rand3, rand1], axis=0)
    rand32 = np.concatenate([rand3, rand2], axis=0)

    x = np.stack([
        rand13,
        rand23,
        rand31,
        rand32,
        np.random.rand(8) * n + n2,
        np.random.rand(8) * n + n1
    ]).T
    xv = x / 255.0

    d_dv = squareform(pdist(xv, 'sqeuclidean'))

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 7)
    for i in range(8):
        for j in range(6):
            ax.scatter(i + 1, j + 1, x[8 - i - 1, 6 - j - 1], c='b', marker='s')
    ax.set_box_aspect(1)
    ax.set_title('X')
    plt.show()

    wt, v_v = nndsvd(x.T, 3, 0)
    st, ht = nndsvd(v_v, 2, 0)
    ht = ht.T

    lambda_val = -4
    alphav = 2
    beta = -4
    pv = 2

    w_mat, s_mat, h_mat, _, _, _ = rlomtfag(x.T, wt, st, ht, d_dv, 10 ** lambda_val, 10 ** alphav, 10 ** beta, 100,
                                            10 ** -10, pv, 'sqeuclidean')

    fig, ax = plt.subplots(figsize=(3, 7))
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0, 7)
    # `kkkkk` might be a scaling factor (to be confirmed)
    kkkkk = 0.5 if np.max(w_mat) > 1000 else 5 if np.max(
        w_mat) > 100 else 50 if np.max(w_mat) > 10 else 500 if np.max(w_mat) > 1 else 1000
    for i in range(6):
        for j in range(3):
            ax.scatter(j + 1, i + 1, kkkkk * w_mat[i, j], c='b', marker='s')
    ax.set_box_aspect(1)
    ax.set_title('W')
    plt.show()

    fig, ax = plt.subplots(figsize=(2, 3))
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(0.5, 3.5)
    kkkkkp = 0.1 if np.max(s_mat) > 1000 else 1 if np.max(
        s_mat) > 100 else 10 if np.max(s_mat) > 10 else 100 if np.max(s_mat) > 1 else 1000
    kkkkkn = 0.1 if np.min(s_mat) < -1000 else 1 if np.min(s_mat) < - \
        100 else 10 if np.min(s_mat) < -10 else 100 if np.min(s_mat) < -1 else 1000
    for i in range(3):
        for j in range(2):
            if s_mat[i, j] >= 0:
                plt.scatter(j + 1, i + 1, kkkkkp *
                            s_mat[i, j], c='b', marker='s', alpha=0.5)
            else:
                plt.scatter(j + 1, i + 1, -kkkkkn *
                            s_mat[i, j], c='r', marker='s', alpha=0.5)
    plt.box(on=True)
    ax.set_title('S')
    plt.show()

    # scatter plot for H
    fig, ax = plt.subplots()
    plt.axis([0, 9, 0.5, 2.5])
    if np.max(h_mat) > 1000:
        kkkkk = 0.5
    elif np.max(h_mat) > 100:
        kkkkk = 5
    elif np.max(h_mat) > 10:
        kkkkk = 50
    elif np.max(h_mat) > 1:
        kkkkk = 500
    else:
        kkkkk = 1000
    for i in range(8):
        for j in range(2):
            plt.scatter(i + 1, j + 1, kkkkk * h_mat[i, j], c='b', marker='s', alpha=0.5)
    plt.box(on=True)
    ax.set_title('H')
    plt.show()


def robust_ability():
    n = 20
    n_abnormal = 5
    k = 1 / 4
    bias = 0.25

    x = np.linspace(1, 10, n)
    x = x + bias * np.random.rand(n)
    y = k * x + bias * np.random.randn(n)

    x_abnormal = 10 * np.random.rand(n_abnormal) + bias * np.random.rand(n_abnormal)
    error = -20
    y_abnormal = k * x_abnormal + bias * np.random.randn(n_abnormal) + error

    x = np.concatenate([x, x_abnormal]).reshape((1, -1))
    y = np.concatenate([y, y_abnormal]).reshape((1, -1))
    xx = np.concatenate([x, y], axis=0)

    r, c = xx.shape
    t = 20
    k_mehx = np.zeros((t, 1))
    # p = 7
    xv = x / 255.0
    d_dv = squareform(pdist(xv.T, metric='sqeuclidean'))
    distance = 'sqeuclidean'

    for i in range(t):
        utv = np.random.rand(r, 2)
        stv = np.random.rand(2, 1)
        vv = np.random.rand(c, 1)
        limiter = 500
        epsilon = 10 ** -10

        lambda_val = -8
        alphav = -2
        beta = -6
        pv = 6

        wv, sv, _, _, _, _ = rlomtfag(
            xx, utv, stv, vv, d_dv, 10 ** lambda_val, 10 ** alphav, 10 ** beta, limiter, epsilon, pv, distance
        )
        wme_hx = wv @ sv
        k_mehx[i] = wme_hx[1] / wme_hx[0]

    xplot = np.arange(0, 12.5, 0.5)
    plt.plot(xplot, np.mean(k_mehx) * xplot, '--r', linewidth=2)
    plt.scatter(x, y, s=100, c='k', marker='o')
    plt.scatter(x_abnormal, y_abnormal, s=100, c='b', marker='o')
    plt.legend(['curve fitting by RLOMTFAG',
                'real data', 'noise data'], fontsize=12)
    plt.grid(True)
    plt.box(True)
    plt.show()


if __name__ == '__main__':
    # robust_ability()
    # adaptive_learning_ability()
    # collaborative_ability()
    cluster_ability()
    print('Done!')
