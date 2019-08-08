"""Frank-Wolfe and related algorithms."""
from copt import utils
import numpy as np
from scipy import optimize
from scipy import sparse
from scipy.sparse import linalg as splinalg
from sklearn.utils.extmath import safe_sparse_dot
from tqdm import trange


def minimize_frank_wolfe(f_grad,
                         x0,
                         lmo,
                         step_size=None,
                         lipschitz=None,
                         max_iter=200,
                         tol=1e-12,
                         callback=None,
                         verbose=0):
  r"""Frank-Wolfe algorithm.

  This method for optimization problems of the form

  .. math::
      \argmin_{\bs{x} \in \mathcal{D}} f(\bs{x})

  where f is a differentiable function for which we have access to its
  gradient and D is a compact set for which we have access to its
  linear minimization oracle (lmo), i.e., a routine that given a vector
  :math:`\bs{u}` returns a solution to

  .. math::
      \argmin_{\bs{x} \in D}\, \langle\bs{u}, \bs{x}\rangle

  Args:
    f_grad: callable
        Takes as input the current iterate (a vector of same size as x0) and
        returns the function value and gradient of the objective function.
        It should accept the optional argument return_gradient, and when False
        it should return only the function value.

    x0 : array-like
        Initial guess for solution.

    lmo: callable
        Takes as input a vector u of same size as x0 and returns a solution to
        the linear minimization oracle (defined above).

    step_size: None or "adaptive" or "adaptive2" or callable
        Step-size step_size to use. If None is used and keyword lipschitz
        is not given or None, then it will use a decreasing step-size of the
        form 2/(k+1) (described in [1]). If None is used and keyword lipschitz
        is not None, then it will use the Demyanov-Rubinov step-size step_size
        (variant 1 in [2]).

    lipschitz: None or float.
        Estimate for the Lipschitz constant of the gradient.

    max_iter: integer

    tol: float

    callback: callable

    verbose: int


  Returns:
    res : scipy.optimize.OptimizeResult
        The optimization result represented as a
        ``scipy.optimize.OptimizeResult`` object. Important attributes are:
        ``x`` the solution array, ``success`` a Boolean flag indicating if
        the optimizer exited successfully and ``message`` which describes
        the cause of the termination. See `scipy.optimize.OptimizeResult`
        for a description of other attributes.


  References:
    [1] Jaggi, Martin. `"Revisiting Frank-Wolfe: Projection-Free Sparse Convex
    Optimization." <http://proceedings.mlr.press/v28/jaggi13-supp.pdf>`_ ICML
    2013.

    [2] Pedregosa, Fabian `"Notes on the Frank-Wolfe Algorithm"
    <http://fa.bianp.net/blog/2018/notes-on-the-frank-wolfe-algorithm-part-i/>`_,
    2018

    [3] Pedregosa, Fabian, Armin Askari, Geoffrey Negiar, and Martin Jaggi.
    `"Step-Size Adaptivity in Projection-Free Optimization."
    <https://arxiv.org/pdf/1806.05123.pdf>`_ arXiv:1806.05123 (2018).
  """
  x0 = sparse.csr_matrix(x0).T
  if tol < 0:
    raise ValueError("Tol must be non-negative")
  x = x0.copy()
  if lipschitz is not None:
    lipschitz_t = lipschitz

  pbar = trange(max_iter, disable=(verbose == 0))
  f_t, grad = f_grad(x)

  it = 0
  for it in pbar:
    s_t = lmo(-grad)
    d_t = s_t - x

    g_t = -safe_sparse_dot(d_t.T, grad)
    assert len(g_t) == 1
    if sparse.issparse(g_t):
      g_t = g_t[0, 0]
    else:
      g_t = g_t[0]
    if g_t <= tol:
      break
    d2_t = splinalg.norm(d_t)**2
    if hasattr(step_size, "__call__"):
      cur_step_size = step_size(locals())
      f_next, grad_next = f_grad(x + cur_step_size * d_t)
    elif step_size == "adaptive":
      ratio_decrease = 0.999
      ratio_increase = 2
      for i in range(max_iter):
        cur_step_size = min(g_t / (d2_t * lipschitz_t), 1)
        rhs = f_t - cur_step_size * g_t + 0.5 * (cur_step_size**2) * lipschitz_t * d2_t
        f_next, grad_next = f_grad(x + cur_step_size * d_t)
        if f_next <= rhs + 1e-6:
          if i == 0:
            lipschitz_t *= ratio_decrease
          break
        else:
          lipschitz_t *= ratio_increase
    elif step_size == "adaptive2":
      rho = 0.2
      for i in range(max_iter):
        cur_step_size = min(g_t / (d2_t * lipschitz_t), 1)
        f_next, grad_next = f_grad(x + cur_step_size * d_t)
        if (f_next - f_t) / cur_step_size < - g_t / 2:
          # we can decrease the Lipschitz / increase the step-size
          lipschitz_t /= 1.5
          continue
        if (f_next - f_t) / cur_step_size >  - rho * g_t / 2:
          lipschitz_t *= 2.
          continue
        break
    elif step_size == "adaptive3":
      rho = 0.9
      for i in range(max_iter):
        cur_step_size = min(g_t / (d2_t * lipschitz_t), 1)
        f_next, grad_next = f_grad(x + cur_step_size * d_t)
        if (f_next - f_t) / cur_step_size > - rho * g_t / 2:
          # sufficient decrease not met, increase Lipchitz constant
          lipschitz_t *= 2
          continue
        if (f_next - f_t) / cur_step_size <= (rho / 2 - 1) * g_t:
          # there's sufficient decrease but the quadratic approximation is not
          # good. We can decrease the Lipschitz / increase the step-size
          lipschitz_t /= 1.5
          continue
        break
      else:
        raise ValueError('Exhausted line search iterations in minimize_frank_wolfe')
    elif step_size == "adaptive4":
      sigma = 0.7
      rho = 0.5
      for i in range(max_iter):
        cur_step_size = min(g_t / (d2_t * lipschitz_t), 1)
        f_next, grad_next = f_grad(x + cur_step_size * d_t)
        if (f_next - f_t) / cur_step_size < - sigma * g_t / 2:
          # we can decrease the Lipschitz / increase the step-size
          lipschitz_t /= 1.5
          continue
        if (f_next - f_t) / cur_step_size > - rho * g_t / 2:
          lipschitz_t *= 2.
          continue
        break
    elif step_size == "adaptive5":
      sigma = 0.9
      rho = 0.4
      eps = .3
      K = 2 * lipschitz_t * d_t_norm / g_t
      M = max((K + sigma) / (K + rho), 1.)
      tau = M * (1 + eps)
      eta = (1 - eps) / M

#       print("Let's check multipliers")
#       print(tau)
#       print(eta)

      for i in range(max_iter):
        cur_step_size = min(g_t / (d2_t * lipschitz_t), 1)
        f_next, grad_next = f_grad(x + cur_step_size * d_t)
        if (f_next - f_t) / cur_step_size < - sigma * g_t / 2:
          # we can decrease the Lipschitz / increase the step-size
          lipschitz_t *= eta
          continue
        if (f_next - f_t) / cur_step_size > - rho * g_t / 2:
          lipschitz_t *= tau
          continue
        break
      else:
        raise ValueError('Exhausted line search iterations in minimize_frank_wolfe')
    elif step_size is None:
      # if we don't know the Lipschitz constant, the best we can do is the 2/(k+2) step-size
      if lipschitz_t is None:
        cur_step_size = 2. / (it+2)
        f_next, grad_next = f_grad(x + cur_step_size * d_t)
      else:
        # this is the case in which we know the Lipschitz constant
        cur_step_size = min(g_t / (d2_t * lipschitz_t), 1)
        f_next, grad_next = f_grad(x + cur_step_size * d_t)
    else:
      raise ValueError("Invalid option step_size=%s" % step_size)
    if callback is not None:
      callback(locals())
    x += cur_step_size * d_t
    pbar.set_postfix(tol=g_t, iter=it, L_t=lipschitz_t)

    f_t, grad = f_next, grad_next
  if callback is not None:
    callback(locals())
  pbar.close()
  x_final = x.toarray().ravel()
  return optimize.OptimizeResult(x=x_final, nit=it, certificate=g_t)


@utils.njit
def max_active(grad, active_set, n_features, include_zero=True):
  """Find the index that most correlates with the gradient."""
  max_grad_active = -np.inf
  max_grad_active_idx = -1
  for j in range(n_features):
    if active_set[j] > 0:
      if grad[j] > max_grad_active:
        max_grad_active = grad[j]
        max_grad_active_idx = j
  for j in range(n_features, 2 * n_features):
    if active_set[j] > 0:
      if -grad[j % n_features] > max_grad_active:
        max_grad_active = -grad[j % n_features]
        max_grad_active_idx = j
  if include_zero:
    if max_grad_active < 0 and active_set[2 * n_features]:
      max_grad_active = 0.
      max_grad_active_idx = 2 * n_features
  return max_grad_active, max_grad_active_idx


def minimize_pairwise_frank_wolfe_l1(f_grad,
                    alpha,
                    n_features,
                    lipschitz=None,
                    max_iter=1000,
                    tol=1e-12,
                    backtracking=True,
                    callback=None,
                    verbose=0):
  """Pairwise FW on the L1 ball.

.. warning::
    This feature is experimental, API is likely to change.

    """

  x = np.zeros(n_features)
  if lipschitz is None:
    lipschitz_t = utils.init_lipschitz(f_grad, x)
  else:
    lipschitz_t = lipschitz

  active_set = np.zeros(2 * n_features + 1)
  active_set[2 * n_features] = 1.
  all_lipschitz = []
  num_bad_steps = 0

  # do a first FW step to
  f_t, grad = f_grad(x)

  pbar = trange(max_iter, disable=(verbose == 0))
  it = 0
  for it in pbar:

    # FW oracle
    idx_oracle = np.argmax(np.abs(grad))
    if grad[idx_oracle] > 0:
      idx_oracle += n_features
    mag_oracle = alpha * np.sign(-grad[idx_oracle % n_features])

    # Away Oracle
    _, idx_oracle_away = max_active(
        grad, active_set, n_features, include_zero=False)

    mag_away = alpha * np.sign(float(n_features - idx_oracle_away))

    is_away_zero = False
    if idx_oracle_away < 0 or active_set[2 * n_features] > 0 and grad[
        idx_oracle_away % n_features] * mag_away < 0:
      is_away_zero = True
      mag_away = 0.
      gamma_max = active_set[2 * n_features]
    else:
      assert grad[idx_oracle_away % n_features] * mag_away > grad.dot(x) - 1e-3
      gamma_max = active_set[idx_oracle_away]

    if gamma_max <= 0:
      pbar.close()
      raise ValueError

    g_t = grad[idx_oracle_away % n_features] * mag_away - \
          grad[idx_oracle % n_features] * mag_oracle
    if g_t <= tol:
      break

    d2_t = 2 * (alpha**2)
    if backtracking:
      # because of the specific form of the update
      # we can achieve some extra efficiency this way
      for i in range(100):
        x_next = x.copy()
        step_size = min(g_t / (d2_t * lipschitz_t), gamma_max)

        x_next[idx_oracle % n_features] += step_size * mag_oracle
        x_next[idx_oracle_away % n_features] -= step_size * mag_away
        f_next, grad_next = f_grad(x_next)
        if step_size < 1e-7:
          break
        elif f_next - f_t <= -g_t * step_size + 0.5 * (step_size**
                                                       2) * lipschitz_t * d2_t:
          if i == 0:
            lipschitz_t *= 0.999
          break
        else:
          lipschitz_t *= 2
      # import pdb; pdb.set_trace()
    else:
      x_next = x.copy()
      step_size = min(g_t / (d2_t * lipschitz_t), gamma_max)
      x_next[idx_oracle %
             n_features] = x[idx_oracle % n_features] + step_size * mag_oracle
      x_next[idx_oracle_away %
             n_features] = x[idx_oracle_away %
                             n_features] - step_size * mag_away
      f_next, grad_next = f_grad(x_next)

    if lipschitz_t >= 1e10:
      raise ValueError
    # was it a drop step?
    # x_t[idx_oracle] += step_size * mag_oracle
    x = x_next
    active_set[idx_oracle] += step_size
    if is_away_zero:
      active_set[2 * n_features] -= step_size
    else:
      active_set[idx_oracle_away] -= step_size
    if active_set[idx_oracle_away] < 0:
      raise ValueError
    if active_set[idx_oracle] > 1:
      raise ValueError

    f_t, grad = f_next, grad_next

    if gamma_max < 1 and step_size == gamma_max:
      num_bad_steps += 1

    if it % 100 == 0:
      all_lipschitz.append(lipschitz_t)
    pbar.set_postfix(
        tol=g_t,
        gmax=gamma_max,
        gamma=step_size,
        L_t_mean=np.mean(all_lipschitz),
        L_t=lipschitz_t,
        bad_steps_quot=(num_bad_steps) / (it + 1))

    if callback is not None:
      callback(locals())

  if callback is not None:
    callback(locals())
  pbar.close()
  return optimize.OptimizeResult(x=x, nit=it, certificate=g_t)
