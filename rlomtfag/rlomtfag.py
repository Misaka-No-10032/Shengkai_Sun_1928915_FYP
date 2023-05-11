import numpy as np
from .utils.evaluation import eval_results
from scipy.sparse import diags, csc_matrix
from scipy.spatial.distance import pdist, squareform
from .utils.mat_utils import e_proj_simplex, l2_distance_1
from .utils.update_params import update_ag_mean, update_u_rlomtfag, update_s_rlomtfag, update_v_rlomtfag


def rlomtfag(x_mat, u0_mat, s0_mat, v0_mat, d_dv_mat, lambda_val, alphav, beta, limiter, epsilon, kv,
             distance_metric='sqeuclidean', t_label=None):
    if t_label is None:
        t_label = []
        time = 0
    else:
        time = 1
    _, num_cols = x_mat.shape
    gv_mat, gamma, _ = update_ag_mean(d_dv_mat, kv)
    dv_mat = diags(np.sum(gv_mat, axis=1))
    ggv_mat = csc_matrix((num_cols, num_cols))
    gv_mat = csc_matrix(gv_mat)
    if time == 1:
        difference = dv_mat - gv_mat
    obj = []
    result = []
    for i in range(limiter):
        y_mat = x_mat - u0_mat @ s0_mat @ v0_mat.T
        ev_mat = np.sqrt(np.sum(y_mat ** 2, axis=0))
        w_mat = np.maximum(2 * ev_mat, epsilon)
        w_mat = diags(1 / w_mat)
        if time == 1:
            g = beta * np.linalg.norm(gv_mat + d_dv_mat / (2 * gamma), 'fro') ** 2
            hh_mat = diags(np.sum(v0_mat, axis=0))
            s_vt = s0_mat @ v0_mat.T
            xt_u = x_mat.T @ u0_mat
            r = u0_mat.shape[1]
            gamma = s_vt @ w_mat @ xt_u - s_vt @ w_mat @ s_vt.T - lambda_val * (
                        s0_mat @ hh_mat @ s0_mat.T - s_vt @ xt_u)
            part1 = np.sum(ev_mat) + alphav * np.trace(v0_mat.T @ difference @ v0_mat)
            part2 = 0
            u_s = u0_mat @ s0_mat
            for j in range(num_cols):
                part2 += np.trace((x_mat[:, j] * np.sum(v0_mat[j, :])
                                   ).reshape(-1, 1) * x_mat[:, j].T)
            part2 += np.trace(u_s @ hh_mat @ u_s.T) - 2 * np.trace(x_mat @ v0_mat @ u_s.T)
            part2 *= lambda_val
            obj.append(part1 + part2 + g +
                       np.trace(gamma @ (u0_mat.T @ u0_mat - np.eye(r))))
            accuracy = 0
            mi_hat = 0
            for k in range(10):
                a, b = eval_results(v0_mat.T, t_label)
                accuracy += a
                mi_hat += b
            result.append([accuracy / 10, mi_hat / 10])

        v1_mat = update_v_rlomtfag(x_mat, u0_mat, s0_mat, v0_mat, w_mat, lambda_val,
                               alphav, gv_mat, dv_mat, epsilon)
        if np.linalg.norm(v1_mat - v0_mat, 'fro') ** 2 < 1e-5:
            v0_mat = v1_mat
            break
        v0_mat = v1_mat
        s0_mat = update_s_rlomtfag(x_mat, u0_mat, v0_mat, w_mat, lambda_val)
        u0_mat = update_u_rlomtfag(x_mat, u0_mat, v0_mat, s0_mat, w_mat, lambda_val)
        if np.linalg.norm(ggv_mat.toarray() - gv_mat.toarray(), 'fro') ** 2 > 1e-3:
            ggv_mat = gv_mat
            v_v = v0_mat / np.max(np.max(v0_mat))
            if distance_metric == 'sqeuclidean':
                dist_v = l2_distance_1(v_v.T, v_v.T)
            elif distance_metric == 'cosine':
                dist_v = squareform(pdist(v_v, metric='cosine'))
            else:
                raise TypeError(f'Unknown distance metric: {distance_metric}')
            gv_mat = np.zeros((num_cols, num_cols))
            for j in range(num_cols):
                idx_a0 = np.arange(num_cols)
                dfi = dist_v[j, idx_a0]
                dxi = d_dv_mat[j, idx_a0]
                ad = -(dxi + beta * dfi) / (2 * gamma)
                gv_mat[j, idx_a0] = e_proj_simplex(ad)
            gv_mat = (gv_mat + gv_mat.T) / 2
            dv_mat = diags(np.sum(gv_mat, axis=1))
            gv_mat = csc_matrix(gv_mat)
            dv_mat = csc_matrix(dv_mat)

    return u0_mat, s0_mat, v0_mat, gv_mat, obj, result


if __name__ == '__main__':
    pass
