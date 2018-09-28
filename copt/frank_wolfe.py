import numpy as np
from numba import njit
from scipy import sparse, optimize
from tqdm import trange
from scipy.sparse import linalg as splinalg
from scipy.stats import hmean
from sklearn.utils.extmath import safe_sparse_dot


def _backtrack(
        f_t, f_grad, x_t, d_t, g_t, L_t,
        gamma_max=1, ratio_increase=2., ratio_decrease=0.999,
        max_iter=100):
    # could be included inside minimize_FW
    d2_t = splinalg.norm(d_t) ** 2
    for i in range(max_iter):
        step_size = min(g_t / (d2_t * L_t), gamma_max)
        rhs = f_t - step_size * g_t + 0.5 * (step_size**2) * L_t * d2_t
        f_next, grad_next = f_grad(x_t + step_size * d_t)
        if f_next <= rhs:
            if i == 0:
                L_t *= ratio_decrease
            break
    else:
        L_t *= ratio_increase
    return step_size, L_t, f_next, grad_next



def minimize_FW(f_grad, lmo, x0, L_t=1, max_iter=1000, tol=1e-12,
          backtracking=True, callback=None, verbose=0):
    """Frank-Wolfe algorithm with L1 ball constraint.
    
    """
    x0 = sparse.csr_matrix(x0).T
    if tol < 0:
        raise ValueError('Tol must be non-negative')
    x_t = x0.copy()
    if callback is not None:
        callback(x_t)
    pbar = trange(max_iter, disable=(verbose == 0))
    f_t, grad = f_grad(x_t)
    L_average = 0.
    for it in pbar:
        s_t = lmo(-grad)
        d_t = s_t - x_t

        g_t = - safe_sparse_dot(d_t.T, grad)
        if sparse.issparse(g_t):
            g_t = g_t[0, 0]
        else:
            g_t = g_t[0]
        if g_t <= tol:
            break
        if backtracking:
            step_size, L_t, f_next, grad_next = _backtrack(
                f_t, f_grad, x_t, d_t, g_t, L_t)
        else:
            d2_t = splinalg.norm(d_t) ** 2
            step_size = min(g_t / (d2_t * L_t), 1)
            f_next, grad_next = f_grad(x_t + step_size * d_t)
        x_t += step_size * d_t
        pbar.set_postfix(tol=g_t, iter=it, step_size=step_size, L_t=L_t, L_average=L_average)

        f_t,  grad = f_next, grad_next
        L_average = L_t / (it + 1) + (it/(it+1)) * L_average
        if callback is not None:
            callback(x_t)
    pbar.close()
    x_final = x_t.toarray().ravel()
    return optimize.OptimizeResult(x=x_final)


@njit
def max_active(grad, active_set, n_features, include_zero=True):
    # find the index that most correlates with the gradient
    max_grad_active = - np.inf
    max_grad_active_idx = -1
    for j in range(n_features):
        if active_set[j] > 0:
            if grad[j] > max_grad_active:
                max_grad_active = grad[j]
                max_grad_active_idx = j
    for j in range(n_features, 2 * n_features):
        if active_set[j] > 0:
            if - grad[j % n_features] > max_grad_active:
                max_grad_active = - grad[j % n_features]
                max_grad_active_idx = j
    if include_zero:
        if max_grad_active < 0 and active_set[2 * n_features]:
            max_grad_active = 0.
            max_grad_active_idx = 2 * n_features
    return max_grad_active, max_grad_active_idx


def minimize_PFW_L1(f_grad, x0, alpha, L_t=1, max_iter=1000, tol=1e-12, ls_strategy='adaptive', callback=None, verbose=0):
    L0 = L_t

    n_features = x0.shape[0]
    active_set = np.zeros(2 * n_features)
    LS_EPS = np.finfo(np.float).eps
    weight_zero = 1.
    all_Lt = []
    num_bad_steps = 0
    
    # do a first FW step to
    _, grad = f_grad(x0)
    idx_oracle = np.argmax(np.abs(grad))
    if grad[idx_oracle] > 0:
        idx_oracle += n_features
    mag_oracle = alpha * np.sign(-grad[idx_oracle % n_features])
    x_t = np.zeros(x0.size)
    step_size =  min(np.abs(mag_oracle) / (alpha * alpha * L_t), 1)
    x_t[idx_oracle % n_features] = mag_oracle
    active_set[idx_oracle] = 1.
    f_t, grad = f_grad(x_t)

    if callback is not None:
        callback(x_t)

    pbar = trange(max_iter, disable=(verbose == 0))
    for it in pbar:
        # FW oracle
        idx_oracle = np.argmax(np.abs(grad))
        if grad[idx_oracle] > 0:
            idx_oracle += n_features
        mag_oracle = alpha * np.sign(-grad[idx_oracle % n_features])

        # Away Oracle
        max_grad_active, idx_oracle_away = max_active(
            grad, active_set, n_features, include_zero=False)

        mag_away = alpha * np.sign(float(n_features - idx_oracle_away))

        gamma_max = active_set[idx_oracle_away]
        if gamma_max <= 0:
            raise ValueError

        g_t = grad[idx_oracle_away % n_features] * mag_away - \
              grad[idx_oracle % n_features] * mag_oracle
        if g_t <= tol:
            break

        if idx_oracle == idx_oracle_away:
            raise ValueError
        else:
            d2_t = 2 * (alpha ** 2)
        if ls_strategy == 'adaptive':
            # because of the specific form of the update
            # we can achieve some extra efficiency this way
            for i in range(100):
                x_next = x_t.copy()
                step_size = min(g_t / (d2_t * L_t), gamma_max)
                rhs = - step_size * g_t + 0.5 * (step_size ** 2) * L_t * d2_t
                rhs2 = - step_size * (g_t - 0.5 * step_size * L_t * d2_t)

                x_next[idx_oracle % n_features] += step_size * mag_oracle
                x_next[idx_oracle_away % n_features] -= step_size * mag_away
                f_next, grad_next = f_grad(x_next)
                if step_size < 1e-7:
                    break
                elif f_next - f_t <= - g_t*step_size + 0.5 * (step_size **2) * L_t * d2_t:
                    if i == 0:
                        L_t *= 0.999
                    break
                else:
                    L_t *= 2
            # import pdb; pdb.set_trace()
        elif ls_strategy == 'Lipschitz':
            x_next = x_t.copy()
            step_size = min(g_t / (d2_t * L_t), gamma_max)
            x_next[idx_oracle % n_features] = x_t[idx_oracle % n_features] + step_size * mag_oracle
            x_next[idx_oracle_away % n_features] = x_t[idx_oracle_away % n_features] - step_size * mag_away
            f_next, grad_next = f_grad(x_next)
        else:
            raise ValueError(ls_strategy)

        if L_t >= 1e10:
            raise ValueError
        # was it a drop step?
        # x_t[idx_oracle] += step_size * mag_oracle
        x_t = x_next
        active_set[idx_oracle] += step_size
        active_set[idx_oracle_away] -= step_size
        if active_set[idx_oracle_away] < 0:
            raise ValueError
        if active_set[idx_oracle] > 1:
            raise ValueError

        f_t, grad = f_next, grad_next
        
        if gamma_max < 1 and step_size == gamma_max:
            num_bad_steps += 1

        if it % 100 == 0:
            all_Lt.append(L_t)
            pbar.set_postfix(
                tol=g_t, Lipschitz=L0, gmax=gamma_max, gamma=step_size, d2t=d2_t,
                L_t_mean=np.mean(all_Lt), L_t=L_t, L_t_harm=hmean(all_Lt),
                bad_steps_quot=(num_bad_steps) / (it+1))

        if callback is not None:
            callback(x_t)
    pbar.close()
    return optimize.OptimizeResult(
        x=x_t)

# 
# def minimize_FW_trace(
#     f_grad, x0, alpha, mat_shape, L_t=1, max_iter=1000, tol=1e-12,
#     backtracking=True, callback=None, verbose=0):
#     """Frank-Wolfe algorithm with trace ball constraint"""
#     if tol < 0:
#         raise ValueError('Tol must be non-negative')
#     x_t = x0.copy()
#     if callback is not None:
#         callback(x_t)
#     pbar = trange(max_iter, disable=(verbose == 0))
#     f_t, grad = f_grad(x_t)
#     L_average = 0.
#     for it in pbar:
#         grad_mat = np.reshape(grad, mat_shape)
#         u, s, vt = splinalg.svds(-grad_mat, k=1, maxiter=1000)
#         s_t = (alpha * u.dot(vt)).ravel()
# 
#         d_t = s_t - x_t
#         g_t = - d_t.T.dot(grad)
#         if g_t <= tol:
#             break
#         if backtracking:
#             step_size, L_t, f_next, grad_next = _backtrack(
#                 f_t, f_grad, x_t, d_t, g_t, L_t)
#         else:
#             d2_t = d_t.dot(d_t)
#             step_size = min(g_t / (d2_t * L_t), 1)
#             f_next, grad_next = f_grad(x_t + step_size * d_t)
#         x_t += step_size * d_t
#         if it % 10 == 0:
#             pbar.set_postfix(tol=g_t, iter=it, step_size=step_size, L_t=L_t, L_average=L_average)
# 
#         f_t,  grad = f_next, grad_next
#         L_average = L_t / (it + 1) + (it/(it+1)) * L_average
#         if callback is not None:
#             callback(x_t)
#     pbar.close()
#     return optimize.OptimizeResult(x=x_t)
