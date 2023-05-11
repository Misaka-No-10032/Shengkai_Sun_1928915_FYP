import numpy as np


def update_ag_mean(d_mat, k):
    d = np.sort(d_mat, axis=1)
    dtk = np.mean(d[:, k + 1])
    gamma = (k * dtk - np.mean(np.sum(d[:, 1:k + 1], axis=1))) / 2
    eta = dtk / (2 * gamma)
    g_mat = np.maximum(eta - d_mat / (2 * gamma), 0)
    return g_mat, gamma, eta


def update_u_rlomtfag(x_mat, u0_mat, v_mat, s_mat, w_mat, lambda_val):
    xp = (np.abs(x_mat) + x_mat) / 2
    xn = (np.abs(x_mat) - x_mat) / 2
    sp = (np.abs(s_mat) + s_mat) / 2
    sn = (np.abs(s_mat) - s_mat) / 2
    xp_v_stp = xp @ v_mat @ sp.T
    xn_v_stn = xn @ v_mat @ sn.T
    xp_v_stn = xp @ v_mat @ sn.T
    xn_v_stp = xn @ v_mat @ sp.T
    xp_y_v_stp = xp @ w_mat @ v_mat @ sp.T
    xn_y_v_stn = xn @ w_mat @ v_mat @ sn.T
    xp_y_v_stn = xp @ w_mat @ v_mat @ sn.T
    xn_y_v_stp = xn @ w_mat @ v_mat @ sp.T
    u1_mat = xp_y_v_stp + xn_y_v_stn
    u2_mat = xp_y_v_stn + xn_y_v_stp
    u3_mat = xp_v_stp + xn_v_stn
    u4_mat = xp_v_stn + xn_v_stp
    first = u1_mat + u0_mat @ u2_mat.T @ u0_mat + lambda_val * (u0_mat @ u4_mat.T @ u0_mat + u3_mat)
    second = u2_mat + u0_mat @ u1_mat.T @ u0_mat + lambda_val * (u0_mat @ u3_mat.T @ u0_mat + u4_mat)
    u_mat = np.multiply(u0_mat, np.divide(first, second))
    return np.asarray(u_mat)


def update_s_rlomtfag(x_mat, u_mat, v_mat, yv_mat, lambda_val):
    d_mat = lambda_val * np.diag(np.sum(v_mat, axis=0))
    ss_mat = u_mat.T @ (x_mat @ (lambda_val * np.eye(x_mat.shape[1]) + yv_mat) @ v_mat) @ np.linalg.inv(
        v_mat.T @ yv_mat @ v_mat + d_mat)
    return np.asarray(ss_mat)


def update_v_rlomtfag(x_mat, u_mat, s_mat, v0_mat, yv_mat, lambda_val, alpha, g_mat, d_mat, epsilon):
    sp = (np.abs(s_mat) + s_mat) / 2
    sn = (np.abs(s_mat) - s_mat) / 2
    k = x_mat.T @ x_mat
    b = np.diag(k)
    n, r = v0_mat.shape
    b_rep = np.tile(b, (r, 1)).T
    xp = (np.abs(x_mat) + x_mat) / 2
    xn = (np.abs(x_mat) - x_mat) / 2
    xpt_u_sp = xp.T @ u_mat @ sp
    xnt_u_sn = xn.T @ u_mat @ sn
    xpt_u_sn = xp.T @ u_mat @ sn
    xnt_u_sp = xn.T @ u_mat @ sp
    spt_sp = sp.T @ sp
    snt_sn = sn.T @ sn
    spt_sn = sp.T @ sn
    yv_v = yv_mat @ v0_mat
    st_s = s_mat.T @ s_mat
    a = np.diag(st_s)
    a_rep = np.tile(a, (n, 1))
    first = 2 * (yv_mat @ (xpt_u_sp + xnt_u_sn) + yv_v @ (spt_sn + spt_sn.T) + alpha * g_mat @ v0_mat + lambda_val * (xpt_u_sp + xnt_u_sn))
    second = 2 * (yv_mat @ (xpt_u_sn + xnt_u_sp) + yv_v @ (spt_sp + snt_sn) + alpha * d_mat @ v0_mat + lambda_val * (
                xpt_u_sn + xnt_u_sp)) + lambda_val * (a_rep + b_rep)
    vs_mat = np.multiply(v0_mat, np.divide(first, np.maximum(second, epsilon)))
    return np.asarray(vs_mat)


if __name__ == '__main__':
    pass
