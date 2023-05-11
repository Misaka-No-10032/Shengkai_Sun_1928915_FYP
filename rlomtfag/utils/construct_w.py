import copy
import warnings
import numpy as np
from mat_utils import eu_dist_2
from scipy.sparse import csc_matrix, lil_matrix
from sklearn.preprocessing import normalize


def construct_w(fea, options={}):
    """
    Construct the affinity matrix W.
    Implemented in the official MATLAB code, but seems not used.
    """
    b_speed = 1

    if 'Metric' in options:
        warnings.warn(
            'This function has been changed and the Metric is no longer be supported.')

    if 'bNormalized' not in options:
        options['bNormalized'] = 0
    # ===============================================
    if 'NeighborMode' not in options:
        options['NeighborMode'] = 'KNN'
    neighbor_mode = options['NeighborMode'].lower()
    if neighbor_mode == 'knn':
        if 'k' not in options:
            options['k'] = 5
    elif neighbor_mode == 'supervised':
        if 'bLDA' not in options:
            options['bLDA'] = 0
        if options['bLDA']:
            options['bSelfConnected'] = 1
        if 'k' not in options:
            options['k'] = 0
        if 'gnd' not in options:
            raise ValueError(
                'gnd should be provided for supervised neighbor mode!')
        if fea.size != 0 and len(options['gnd']) != fea.shape[0]:
            raise ValueError('gnd and fea should have the same length!')
    else:
        raise ValueError('NeighborMode does not exist!')
    # ===============================================
    if 'WeightMode' not in options:
        options['WeightMode'] = 'HeatKernel'

    b_binary = 0
    b_cosine = 0

    weight_mode = options['WeightMode'].lower()
    if weight_mode == 'binary':
        b_binary = 1
    elif weight_mode == 'heatkernel':
        if 't' not in options:
            n_smp = fea.shape[0]
            if n_smp > 3000:
                distance = eu_dist_2(np.random.choice(fea, size=3000, replace=False))
            else:
                distance = eu_dist_2(fea)
            options['t'] = np.mean(distance)
    elif weight_mode == 'cosine':
        b_cosine = 1
    else:
        raise ValueError('WeightMode does not exist!')
    # ===============================================
    if 'bSelfConnected' not in options:
        options['bSelfConnected'] = 0

    # ===============================================
    if 'gnd' in options:
        n_smp = len(options['gnd'])
    else:
        n_smp = fea.shape[0]

    max_m = 62500000  # 500M
    block_size = int(max_m / (n_smp * 3))

    if neighbor_mode == 'supervised':
        labels = np.unique(options['gnd'])
        n_label = len(labels)
        if options['bLDA']:
            g_mat = np.zeros((n_smp, n_smp))
            for idx in range(n_label):
                class_idx = options['gnd'] == labels[idx]
                g_mat[np.outer(class_idx.T, class_idx)] = 1 / np.sum(class_idx)
            w_mat = csc_matrix(g_mat)
            return w_mat
        if weight_mode == 'binary':
            if options['k'] > 0:
                g_mat = np.zeros((n_smp * (options['k'] + 1), 3))
                id_now = 0
                for label in labels:
                    class_idx = np.where(options['gnd'] == label)[0]
                    distance = eu_dist_2(fea[class_idx, :], np.array([]), False)
                    idx = np.argsort(distance, axis=1)[:, :options['k'] + 1]
                    n_smp_class = len(class_idx) * (options['k'] + 1)
                    g_mat[id_now:id_now + n_smp_class, 0] = np.tile(class_idx, (options['k'] + 1))
                    g_mat[id_now:id_now + n_smp_class, 1] = class_idx[idx.reshape(-1)]
                    g_mat[id_now:id_now + n_smp_class, 2] = 1
                    id_now = id_now + n_smp_class
                g_mat = csc_matrix((g_mat[:, 2], (g_mat[:, 0], g_mat[:, 1])),
                                   shape=(n_smp, n_smp))
                g_mat = np.maximum(g_mat, g_mat.T)
            else:
                g_mat = np.zeros((n_smp, n_smp))
                for label in labels:
                    class_idx = np.where(options['gnd'] == label)[0]
                    g_mat[np.outer(class_idx.T, class_idx)] = 1

            if not options['bSelfConnected']:
                np.fill_diagonal(g_mat, 0)

            w_mat = csc_matrix(g_mat)
        elif weight_mode == 'heatkernel':
            if options['k'] > 0:
                g_mat = np.zeros((n_smp * (options['k'] + 1), 3))
                id_now = 0
                for label in labels:
                    class_idx = np.where(options['gnd'] == label)[0]
                    distance = eu_dist_2(fea[class_idx, :], np.array([]), False)
                    idx = np.argsort(distance, axis=1)
                    idx = idx[:, :options['k'] + 1]
                    dump = distance[np.arange(len(class_idx))[:, None], idx]
                    dump = np.exp(-dump / (2 * options['t'] ** 2))
                    n_smp_class = len(class_idx) * (options['k'] + 1)
                    g_mat[id_now:id_now + n_smp_class, 0] = np.tile(class_idx, (options['k'] + 1))
                    g_mat[id_now:id_now + n_smp_class, 1] = class_idx[idx].flatten()
                    g_mat[id_now:id_now + n_smp_class, 2] = dump.flatten()
                    id_now += n_smp_class
                g_mat = csc_matrix((g_mat[:, 2], (g_mat[:, 0], g_mat[:, 1])),
                                   shape=(n_smp, n_smp))
            else:
                g_mat = np.zeros((n_smp, n_smp))
                for label in labels:
                    class_idx = np.where(options['gnd'] == label)[0]
                    distance = eu_dist_2(fea[class_idx, :], np.array([]), False)
                    dump = np.exp(-distance / (2 * options['t'] ** 2))
                    g_mat[np.outer(class_idx.T, class_idx)] = dump
            if not options['bSelfConnected']:
                np.fill_diagonal(g_mat, 0)
            w_mat = csc_matrix(np.maximum(g_mat, g_mat.T))
        elif weight_mode == 'cosine':
            if not options['bNormalized']:
                fea = normalize(fea)
            if options['k'] > 0:
                g_mat = np.zeros((n_smp * (options['k'] + 1), 3))
                id_now = 0
                for label in labels:
                    class_idx = np.where(options['gnd'] == label)[0]
                    distance = np.dot(fea[class_idx, :], fea[class_idx, :].T)
                    dump, idx = np.sort(-distance, axis=1), np.argsort(-distance, axis=1)
                    idx = idx[:, :options['k'] + 1]
                    dump = -dump[:, :options['k'] + 1]
                    n_smp_class = len(class_idx) * (options['k'] + 1)
                    g_mat[id_now:id_now + n_smp_class,
                    0] = np.tile(class_idx, options['k'] + 1)
                    g_mat[id_now:id_now + n_smp_class, 1] = class_idx[idx].flatten()
                    g_mat[id_now:id_now + n_smp_class, 2] = dump.flatten()
                    id_now += n_smp_class
                g_mat = csc_matrix((g_mat[:, 2], (g_mat[:, 0], g_mat[:, 1])),
                                   shape=(n_smp, n_smp))
            else:
                g_mat = np.zeros((n_smp, n_smp))
                for label in labels:
                    class_idx = np.where(options['gnd'] == label)[0]
                    g_mat[np.outer(class_idx.T, class_idx)] = np.dot(
                        fea[class_idx, :], fea[class_idx, :].T)

            if not options['bSelfConnected']:
                np.fill_diagonal(g_mat, 0)

            w_mat = csc_matrix(np.maximum(g_mat, g_mat.T))
        else:
            raise ValueError('WeightMode does not exist!')
        return w_mat
    if b_cosine and not options['bNormalized']:
        norm_fea = normalize(fea)
    if neighbor_mode == 'knn' and options['k'] > 0:
        if not (b_cosine and options['bNormalized']):
            g_mat = np.zeros((n_smp * (options['k'] + 1), 3))

            for i in range(np.ceil(n_smp / block_size).astype(int)):
                if i == np.ceil(n_smp / block_size) - 1:
                    smp_idx = np.arange(int(i * block_size), n_smp)
                    dist = eu_dist_2(fea[smp_idx, :], fea, False)

                    if b_speed:
                        n_smp_now = len(smp_idx)
                        dump = np.zeros((n_smp_now, options['k'] + 1))
                        idx = copy.deepcopy(dump)
                        for j in range(options['k'] + 1):
                            dump[:, j], idx[:, j] = np.min(
                                dist, axis=1), np.argmin(dist, axis=1)
                            temp = idx[:, j] * n_smp_now + \
                                   np.arange(n_smp_now)[:, np.newaxis]
                            temp = temp.diagonal().astype(int)
                            dist.ravel()[temp] = 1e100
                    else:
                        dump, idx = np.sort(
                            dist, axis=1), np.argsort(dist, axis=1)
                        idx = idx[:, :options['k'] + 1]
                        dump = dump[:, :options['k'] + 1]

                    if not b_binary:
                        if b_cosine:
                            dist = norm_fea[smp_idx, :] @ norm_fea.T
                            dist = dist.toarray()
                            lin_idx = np.arange(idx.shape[0])[:, np.newaxis]
                            # lin_idx = lin_idx.repeat(idx.shape[1], axis=1)
                            dump = dist[np.ravel_multi_index(
                                (lin_idx, idx), dist.shape)]
                        else:
                            dump = np.exp(-dump / (2 * options['t'] ** 2))
                    g_mat[i * block_size * (options['k'] + 1):n_smp * (options['k'] + 1), 0] = np.tile(smp_idx, options['k'] + 1)
                    g_mat[i * block_size * (options['k'] + 1):n_smp * (options['k'] + 1), 1] = idx.flatten()

                    if not b_binary:
                        g_mat[i * block_size * (options['k'] + 1):n_smp * (options['k'] + 1), 2] = dump.flatten()
                    else:
                        g_mat[i * block_size * (options['k'] + 1):n_smp * (options['k'] + 1), 2] = 1
                else:
                    smp_idx = np.arange(i * block_size, (i + 1) * block_size)
                    dist = eu_dist_2(fea[smp_idx, :], fea, False)
                    if b_speed:
                        n_smp_now = len(smp_idx)
                        dump = np.zeros((n_smp_now, options['k'] + 1))
                        idx = copy.deepcopy(dump)
                        for j in range(options['k'] + 1):
                            dump[:, j], idx[:, j] = np.min(
                                dist, axis=1), np.argmin(dist, axis=1)
                            temp = idx[:, j] * n_smp_now + np.arange(n_smp_now)[:, np.newaxis]
                            temp = temp.diagonal().astype(int)
                            dist.ravel()[temp] = 1e100
                    else:
                        dump, idx = np.sort(
                            dist, axis=1), np.argsort(dist, axis=1)
                        idx = idx[:, :options['k'] + 1]
                        dump = dump[:, :options['k'] + 1]

                    if not b_binary:
                        if b_cosine:
                            dist = norm_fea[smp_idx, :] @ norm_fea.T
                            dist = dist.toarray()
                            lin_idx = np.arange(idx.shape[0])[:, np.newaxis]
                            # lin_idx = lin_idx.repeat(idx.shape[1], axis=1)
                            dump = dist[np.ravel_multi_index(
                                (lin_idx, idx), dist.shape)]
                        else:
                            dump = np.exp(-dump / (2 * options['t'] ** 2))

                    g_mat[i * block_size * (options['k'] + 1): (i + 1) * block_size *
                                                               (options['k'] + 1), 0] = np.tile(smp_idx,
                                                                                                options['k'] + 1)
                    g_mat[i * block_size * (options['k'] + 1): (i + 1) * block_size *
                                                               (options['k'] + 1), 1] = idx.flatten()
                    if not b_binary:
                        g_mat[i * block_size * (options['k'] + 1): (i + 1) * block_size *
                                                                   (options['k'] + 1), 2] = dump.flatten()
                    else:
                        g_mat[i * block_size * (options['k'] + 1): (i + 1) * block_size * (options['k'] + 1), 2] = 1

            w_mat = csc_matrix((g_mat[:, 2], (g_mat[:, 0], g_mat[:, 1])), shape=(n_smp, n_smp))
        else:
            g_mat = np.zeros((n_smp * (options['k'] + 1), 3))
            for i in range(np.ceil(n_smp / block_size).astype(int)):
                if i == np.ceil(n_smp / block_size) - 1:
                    smp_idx = np.arange(int(i * block_size), n_smp)
                    dist = norm_fea[smp_idx, :] @ norm_fea.T
                    dist = dist.toarray()

                    if b_speed:
                        n_smp_now = len(smp_idx)
                        dump = np.zeros((n_smp_now, options['k'] + 1))
                        idx = copy.deepcopy(dump)
                        for j in range(options['k'] + 1):
                            dump[:, j], idx[:, j] = np.min(
                                dist, axis=1), np.argmin(dist, axis=1)
                            temp = idx[:, j] * n_smp_now + \
                                   np.arange(n_smp_now)[:, np.newaxis]
                            temp = temp.diagonal().astype(int)
                            dist.ravel()[temp] = 0
                    else:
                        dump, idx = np.sort(-dist,
                                            axis=1), np.argsort(-dist, axis=1)
                        idx = idx[:, :options['k'] + 1]
                        dump = -dump[:, :options['k'] + 1]

                    g_mat[i * block_size * (options['k'] + 1):n_smp * (options['k'] + 1), 0] = np.tile(smp_idx, options['k'] + 1)
                    g_mat[i * block_size * (options['k'] + 1):n_smp * (options['k'] + 1), 1] = idx.flatten()
                    g_mat[i * block_size * (options['k'] + 1):n_smp * (options['k'] + 1), 2] = dump.flatten()
                else:
                    smp_idx = np.arange(i * block_size, (i + 1) * block_size)
                    dist = fea[smp_idx, :] @ fea.T
                    dist = dist.toarray()

                    if b_speed:
                        n_smp_now = len(smp_idx)
                        dump = np.zeros((n_smp_now, options['k'] + 1))
                        idx = copy.deepcopy(dump)
                        for j in range(options['k'] + 1):
                            dump[:, j], idx[:, j] = np.min(
                                dist, axis=1), np.argmin(dist, axis=1)
                            temp = idx[:, j] * n_smp_now + \
                                   np.arange(n_smp_now)[:, np.newaxis]
                            temp = temp.diagonal().astype(int)
                            dist.ravel()[temp] = 0
                    else:
                        dump, idx = np.sort(-dist,
                                            axis=1), np.argsort(-dist, axis=1)
                        idx = idx[:, :options['k'] + 1]
                        dump = -dump[:, :options['k'] + 1]

                    g_mat[i * block_size * (options['k'] + 1): (i + 1) * block_size *
                                                               (options['k'] + 1), 0] = np.tile(smp_idx,
                                                                                                options['k'] + 1)
                    g_mat[i * block_size * (options['k'] + 1): (i + 1) * block_size *
                                                               (options['k'] + 1), 1] = idx.flatten()
                    g_mat[i * block_size * (options['k'] + 1): (i + 1) * block_size *
                                                               (options['k'] + 1), 2] = dump.flatten()

            w_mat = csc_matrix((g_mat[:, 2], (g_mat[:, 0], g_mat[:, 1])), shape=(n_smp, n_smp))

        if b_binary:
            w_mat[w_mat != 0] = 1

        if 'bSemiSupervised' in options and options['bSemiSupervised']:
            tmp_gnd = options['gnd'][options['semiSplit']]
            labels = np.unique(tmp_gnd)
            n_label = len(labels)
            g_mat = np.zeros(
                (np.sum(options['semiSplit']), np.sum(options['semiSplit'])))

            for label in labels:
                class_idx = np.where(tmp_gnd == label)[0]
                g_mat[np.outer(class_idx.T, class_idx)] = 1
            w_sup = csc_matrix(g_mat)

            if 'SameCategoryWeight' not in options:
                options['SameCategoryWeight'] = 1

            w_mat[options['semiSplit'], options['semiSplit']] = (
                                                                        w_sup > 0) * options['SameCategoryWeight']

        if not options['bSelfConnected']:
            w_mat = lil_matrix(w_mat)
            w_mat.setdiag(0)

        if not ('bTrueKNN' in options and options['bTrueKNN']):
            w_mat = np.maximum(w_mat.toarray(), w_mat.toarray().T)

        return w_mat
    if weight_mode == 'binary':
        raise ValueError(
            'Binary weight can not be used for complete graph!')
    elif weight_mode == 'heatkernel':
        w_mat = eu_dist_2(fea, np.array([]), False)
        w_mat = np.exp(-w_mat / (2 * options['t'] ** 2))
    elif weight_mode == 'cosine':
        w_mat = np.dot(norm_fea, norm_fea.T).full()
    else:
        raise ValueError('WeightMode does not exist!')

    if not options['bSelfConnected']:
        w_mat = lil_matrix(w_mat)
        w_mat.setdiag(0)

    w_mat = np.maximum(w_mat, w_mat.T)
    return w_mat


if __name__ == '__main__':
    pass
