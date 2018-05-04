import numpy as np
from numba import njit
from scipy import sparse
from tqdm import trange
from scipy.sparse import linalg as splinalg
from scipy.stats import hmean

def backtrack(
        f_t, f_grad, x_t, d_t, g_t, L_t,
        gamma_max=1, ratio_increase=2., ratio_decrease=0.999,
        max_iter=100):
    d2_t = d_t.T.dot(d_t)
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


# do line search by bysection
def approx_ls(f_t, f_grad, x_t, d_t, g_t, L_t,
        gamma_max=1, ratio_increase=2., ratio_decrease=0.99,
        max_iter=20):
    # approximate line search
    def obj(gamma):
        f_next, grad = f_grad(x_t + gamma * d_t)
        grad_gamma = d_t.T.dot(grad)[0]
        return f_next, grad_gamma
    lbracket = 0
    grad_lbracket = obj(lbracket)[1]
    rbracket = gamma_max
    grad_rbracket = obj(rbracket)[1]
    for _ in range(max_iter):
        assert grad_lbracket * grad_rbracket <= 0
        c = (lbracket + rbracket) / 2.
        fc, grad_c = obj(c)
        if grad_c * grad_lbracket >= 0:
            lbracket = c
            grad_lbracket = grad_c
        else:
            rbracket = c
            grad_rbracket = grad_c
    out = (lbracket + rbracket) / 2.
    return out


def minimize_FW_L1(f_grad, x0, alpha, L_t=1, max_iter=100, tol=1e-12,
          ls_strategy='adaptive', callback=None):
    x_t = x0.copy()
    if callback is not None:
        callback(x_t)
    pbar = trange(max_iter)
    f_t, grad = f_grad(x_t)
    L_average = 0.
    for it in pbar:
        idx_oracle = np.argmax(np.abs(grad))
        mag_oracle = alpha * np.sign(-grad[idx_oracle])
        d_t = - x_t.copy()
        d_t[idx_oracle] += mag_oracle
        g_t = - d_t.T.dot(grad).ravel()[0]
        if g_t <= tol:
            break
        if ls_strategy == 'adaptive':
            step_size, L_t, f_next, grad_next = backtrack(
                f_t, f_grad, x_t, d_t, g_t, L_t)
        elif ls_strategy == 'Lipschitz':
            step_size = min(g_t / (d_t.T.dot(d_t)[0, 0] * L_t), 1)
            f_next, grad_next = f_grad(x_t + step_size * d_t)
        elif ls_strategy == 'approx_ls':
            step_size = approx_ls(
                f_t, f_grad, x_t, d_t, g_t, L_t)
            f_next, grad_next = f_grad(x_t + step_size * d_t)
        x_t += step_size * d_t
        if it % 10 == 0:
            pbar.set_postfix(tol=g_t, iter=it, step_size=step_size, L_t=L_t, L_average=L_average)

        f_t,  grad = f_next, grad_next
        L_average = L_t / (it + 1) + (it/(it+1)) * L_average
        if callback is not None:
            callback(x_t)
    return x_t




def minimize_FW_trace(A, x0, alpha, L_t=1, max_iter=100, tol=1e-12,
          ls_strategy='adaptive', callback=None):
    x_t = x0.copy()
    if callback is not None:
        callback(x_t)
    pbar = trange(max_iter)

    def func_grad(X, grad_data, A_data, A_indices, A_indptr):
        obj = 0
        for i in range(grad.shape[0]):
            for j in range(A_indptr[i], A_indptr[i + 1]):
                j_idx = A_indices[j]
                grad_data[j] = X[i, j_idx] - A_data[j]
                obj += 0.5 * ((X[i, j_idx] - A_data[j]) ** 2)
        return obj

    grad = A.copy()
    f_t = func_grad(x_t, grad.data, A.data, A.indices, A.indptr)

    for it in pbar:

        u, s, vt = splinalg.svds(grad, k=1)

        d_t = np.outer(u[0], vt[0]) - x_t
        g_t = - d_t.T.dot(grad).ravel()[0]
        1/0
        if g_t <= tol:
            break
        if ls_strategy == 'adaptive':
            step_size, L_t, f_next, grad_next = backtrack(
                f_t, f_grad, x_t, d_t, g_t, L_t)
        elif ls_strategy == 'Lipschitz':
            step_size = min(g_t / (d_t.T.dot(d_t)[0, 0] * L_t), 1)
            f_next, grad_next = f_grad(x_t + step_size * d_t)
        elif ls_strategy == 'approx_ls':
            step_size = approx_ls(
                f_t, f_grad, x_t, d_t, g_t, L_t)
            f_next, grad_next = f_grad(x_t + step_size * d_t)
        x_t += step_size * d_t
        if it % 10 == 0:
            pbar.set_postfix(tol=g_t, iter=it, step_size=step_size)

        f_t,  grad = f_next, grad_next
        if callback is not None:
            callback(x_t)
    return x_t




def minimize_FW_L1_precond(f_grad, x0, alpha, L_t=1, max_iter=100, tol=1e-12,
          ls_strategy='adaptive', callback=None):
    x_t = x0.copy()
    if callback is not None:
        callback(x_t)
    h_diag = np.ones(x0.shape[0])
    n_diag = np.zeros(x0.shape[0])
    pbar = trange(max_iter)
    f_t, grad = f_grad(x_t)
    for it in pbar:
        idx_oracle = np.argmax(np.abs(grad) / h_diag)
        mag_oracle = alpha * np.sign(-grad[idx_oracle])
        d_t = - x_t.copy()
        d_t[idx_oracle] += mag_oracle
        g_t = - d_t.T.dot(grad).ravel()[0]
        if g_t <= tol:
            break
        if ls_strategy == 'adaptive':
            step_size, L_t, f_next, grad_next = backtrack(
                f_t, f_grad, x_t, d_t, g_t, L_t)
        elif ls_strategy == 'Lipschitz':
            step_size = min(g_t / (d_t.T.dot(d_t)[0, 0] * L_t), 1)
        elif ls_strategy == 'approx_ls':
            step_size = approx_ls(
                f_t, f_grad, x_t, d_t, g_t, L_t)
        x_t += step_size * d_t
        if it % 10 == 0:
            pbar.set_postfix(tol=g_t, iter=it, step_size=step_size)

        if callback is not None:
            callback(x_t)

        # estimated diagonal of hessian
        # f_next, grad_next = f_grad(x_t)
        hess_i = np.abs((grad_next - grad)[idx_oracle] / (step_size * d_t[idx_oracle].toarray()))
        n = n_diag[idx_oracle]
        h_diag[idx_oracle] = (n/(n+1.)) * h_diag[idx_oracle] + (1./(n+1)) * hess_i

        f_t, grad = f_next, grad_next
        if it % 100 == 0:
            print(h_diag)

    return x_t

@njit
def max_active(grad, active_set, n_features):
    # find the index that most correlates with the gradient
    max_grad_active = - np.inf
    max_grad_active_idx = -1
    for j in range(n_features):
        if active_set[j]:
            if grad[j] > max_grad_active:
                max_grad_active = grad[j]
                max_grad_active_idx = j
    for j in range(n_features, 2 * n_features):
        if active_set[j]:
            if - grad[j % n_features] > max_grad_active:
                max_grad_active = - grad[j % n_features]
                max_grad_active_idx = j
    if max_grad_active < 0 and active_set[2 * n_features]:
        max_grad_active = 0.
        max_grad_active_idx = 2 * n_features
    return max_grad_active, max_grad_active_idx


def minimize_PFW_L1(f_grad, x0, alpha, L_t=1, max_iter=1000, tol=1e-12,
          ls_strategy='adaptive', callback=None):
    x_t = np.zeros(x0.size)
    L0 = L_t
    if callback is not None:
        callback(x_t)

    n_features = x0.shape[0]
    active_set = np.zeros(2 * n_features + 1, dtype=np.bool)
    active_set[2 * n_features] = True
    # active_set_weights = np.zeros(2 * n_features + 1, dtype=np.float)
    # active_set_weights[-1] = 1.
    LS_EPS = np.finfo(np.float).eps
    weight_zero = 1.
    all_Lt = []

    pbar = trange(max_iter)
    f_t, grad = f_grad(x_t)
    for it in pbar:
        # f_t, grad = f_grad(x_t)
        idx_oracle = np.argmax(np.abs(grad))
        if grad[idx_oracle] > 0:
            idx_oracle += n_features
        mag_oracle = alpha * np.sign(-grad[idx_oracle % n_features])

        max_grad_active, max_grad_active_idx = max_active(
            grad, active_set, n_features)

        mag_away = alpha * np.sign(float(n_features - max_grad_active_idx))

        is_away_zero = (max_grad_active_idx == 2 * n_features)
        if is_away_zero:
            gamma_max = weight_zero
        else:
            gamma_max = np.abs(x_t[max_grad_active_idx % n_features]) / alpha
        assert gamma_max > 0

        g_t = grad[max_grad_active_idx % n_features] * mag_away - \
              grad[idx_oracle % n_features] * mag_oracle
        if g_t <= tol:
            break
        if ls_strategy == 'adaptive':
            if idx_oracle == max_grad_active_idx:
                raise ValueError
            else:
                d2_t = 2 * (alpha ** 2)

            # if g_t >= 1e-6:
            x_next = x_t.copy()
            for i in range(100):
                step_size = min(g_t / (d2_t * L_t), gamma_max)
                rhs = - step_size * g_t + 0.5 * (step_size ** 2) * L_t * d2_t
                rhs2 = - step_size * (g_t - 0.5 * step_size * L_t * d2_t)

                x_next[idx_oracle % n_features] = x_t[idx_oracle % n_features] + step_size * mag_oracle
                if not is_away_zero:
                    x_next[max_grad_active_idx % n_features] = x_t[max_grad_active_idx % n_features] - step_size * mag_away
                f_next, grad_next = f_grad(x_next)
                if step_size < 1e-7:
                    break
                elif (f_next - f_t)/step_size <= -(g_t - 0.5 * step_size * L_t * d2_t) + LS_EPS/step_size:
                    if i == 0:
                        L_t *= 0.999
                    break
                else:
                    if L_t > 2 * L0:
                        raise ValueError
                    L_t *= 2
                    # print(i, L_t, f_next, rhs)
            # else:
            #     step_size = min(g_t / (d2_t * L_t), gamma_max)
            #     rhs = f_t - step_size * g_t + 0.5 * (step_size ** 2) * L_t * d2_t
            #     x_next[idx_oracle % n_features] = x_t[idx_oracle % n_features] + step_size * mag_oracle
            #     if not is_away_zero:
            #         x_next[max_grad_active_idx % n_features] = x_t[max_grad_active_idx % n_features] - step_size * mag_away
            #     f_next, grad_next = f_grad(x_next)
        elif ls_strategy == 'Lipschitz':
            step_size = min(g_t / (d2_t * L_t), gamma_max)
            x_next[idx_oracle % n_features] = x_t[idx_oracle % n_features] + step_size * mag_oracle
            if not is_away_zero:
                x_next[max_grad_active_idx % n_features] = x_t[max_grad_active_idx % n_features] - step_size * mag_away
            f_next, grad_next = f_grad(x_next)

        if L_t >= 1e10:
            raise ValueError
        # was it a drop step?
        # x_t[idx_oracle] += step_size * mag_oracle
        x_t = x_next
        active_set[idx_oracle] = (x_t[idx_oracle % n_features] != 0)

        if it > 0:
            if max_grad_active_idx == 2 * n_features:
                pass
            # x_t[max_grad_active_idx] -= step_size * mag_away
            if is_away_zero:
                weight_zero -= step_size
                if weight_zero == 0:
                    active_set[2 * n_features] = False
            else:
                active_set[max_grad_active_idx] = (x_t[max_grad_active_idx % n_features] != 0)

        f_t, grad = f_next, grad_next

        # x_t += step_size * d_t
        all_Lt.append(L_t)
        if it % 100 == 0:
            pbar.set_postfix(
                tol=g_t, iter=it, Lipschitz=L0, gmax=gamma_max, gamma=step_size, d2t=d2_t,
                L_t_mean=np.mean(all_Lt), L_t=L_t, L_t_harm=hmean(all_Lt))

        # if it > int(3e+06):
        #     import ipdb;
        #     ipdb.set_trace()

        if callback is not None:
            callback(x_t)
    pbar.close()
    return x_t




def minimize_AFW_L1XXX(
    func_grad, x0, alpha, max_iter=100, ratio_increase=2.,
    ratio_decrease=0.9, tol=1e-12, callback=None, L_t=1):
    n_features = x0.shape[0]
    x_t = x0.copy()
    trace = []
    vertices = alpha * np.concatenate([np.eye(n_features), -np.eye(n_features)])
    active_set = set()
    trace.append(func_grad(x_t)[0])
    pbar = trange(max_iter)
    for it in pbar:
        f, grad = func_grad(x_t)
        k = np.argmax(vertices.dot(-grad))
        s_t = vertices[k]
        d_fw = s_t - x_t
        g_t = np.dot(-grad, d_fw)
        fw_step = True
        if len(active_set) > 0:
            for ka in active_set:
                if (-grad).dot(x_t - vertices[ka]) > g_t:
                    fw_step = False
                    d_a = x_t - vertices[ka]
                    alpha_v = np.abs(x_t[ka % n_features]) / alpha
                    gamma_max = alpha_v / (1. - alpha_v)
                    d_t = d_a
                    break
            else:
                d_t = d_fw
                gamma_max = 1.

        else:
            d_t = d_fw
            gamma_max = 1.

        q_t = (-grad).dot(d_t)
        for _ in range(100):
            step_size = min(q_t / (d_t.dot(d_t) * L_t), gamma_max)
            rhs = f - step_size * q_t + 0.5 * (step_size**2) * L_t * d_t.dot(d_t)
            if func_grad(x_t + step_size * d_t)[0] <= rhs + 1e-8:
                L_t *= ratio_decrease
                break
            else:
                L_t *= ratio_increase
                # print(it, d_t.dot(d_t), L_t, q_t, func_grad(x_t + step_size * d_t)[0], rhs)
        x_t += step_size * d_t

        # update active set
        if fw_step:
            if step_size == 1:
                active_set = set([k])
            else:
                # should check if its not in there already
                active_set.add(k)
        else:
            if step_size == gamma_max:
                print('Drop step', it)
                active_set.remove(ka)
            else:
                print('Away step', it)

        if callback is not None:
            callback(x_t)

    return x_t, np.array(trace)
