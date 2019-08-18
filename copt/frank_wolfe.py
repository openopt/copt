"""Frank-Wolfe and related algorithms."""
import warnings
from copt import utils
import numpy as np
from scipy import linalg
from scipy import optimize
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

  Implements the Frank-Wolfe algorithm, see , see :ref:`frank_wolfe` for
  a more detailed description.

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
      Maximum number of iterations.

    tol: float
      Tolerance of the stopping criterion. The algorithm will stop whenever
      the Frank-Wolfe gap is below tol or the maximum number of iterations
      is exceeded.

    callback: callable
      Callback to execute at each iteration. If the callable returns False
      then the algorithm with immediately return.

    verbose: int
      Verbosity level.


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


  Examples
  --------
    * :ref:`sphx_glr_auto_examples_frank_wolfe_plot_fw_stepsize.py`
    * :ref:`sphx_glr_auto_examples_frank_wolfe_plot_fw_vertex_overlap.py`
  """
  x0 = np.asanyarray(x0, dtype=np.float)
  if tol < 0:
    raise ValueError("Tol must be non-negative")
  x = x0.copy()
  if lipschitz is not None:
    lipschitz_t = lipschitz

  pbar = trange(max_iter, disable=(verbose == 0))
  f_t, grad = f_grad(x)
  old_f_t = None

  it = 0
  for it in pbar:
    update_direction, certificate = lmo(-grad, x)

    if certificate <= tol:
      break
    norm_update_direction = linalg.norm(update_direction)**2
    if hasattr(step_size, "__call__"):
      step_size_t = step_size(locals())
      f_next, grad_next = f_grad(x + step_size_t * update_direction)
    elif step_size == "adaptive":
      ratio_decrease = 0.999
      ratio_increase = 2
      for i in range(max_iter):
        step_size_t = min(certificate / (norm_update_direction * lipschitz_t), 1)
        rhs = f_t - step_size_t * certificate + \
          0.5 * (step_size_t**2) * lipschitz_t * norm_update_direction
        f_next, grad_next = f_grad(x + step_size_t * update_direction)
        if f_next <= rhs + 1e-6:
          if i == 0:
            lipschitz_t *= ratio_decrease
          break
        else:
          lipschitz_t *= ratio_increase
    elif step_size == "adaptive2":
      from .line_search import line_search_wolfe1
      out = line_search_wolfe1(
        lambda z: f_grad(z)[0],
        lambda z: f_grad(z)[1], 
        x,
        update_direction,
        gfk=grad, old_fval=f_t,
        old_old_fval=old_f_t,
        amax=1
        )
      step_size_t = out[0]
      if step_size_t is None:
        step_size_t = min(certificate / (norm_update_direction * lipschitz_t), 1)
      f_next = out[3]
      grad_next = out[-1]
    elif step_size == "adaptive2+":
      if lipschitz_t is None:
        raise ValueError
      from .line_search import line_search_wolfe1
      alpha1 = min(certificate / (norm_update_direction * lipschitz_t), 1)
      out = line_search_wolfe1(
        lambda z: f_grad(z)[0],
        lambda z: f_grad(z)[1], 
        x,
        update_direction,
        gfk=grad, old_fval=f_t,
        old_old_fval=old_f_t,
        alpha1=alpha1,
        amax=1
        )
      step_size_t = out[0]
      if step_size_t is None:
        step_size_t = alpha1
      elif step_size_t < 1:
        lipschitz_t = certificate / (norm_update_direction * step_size_t)
      f_next = out[3]
      grad_next = out[-1]
    elif step_size == "adaptive3":
      rho = 0.9
      for i in range(max_iter):
        step_size_t = min(certificate / (norm_update_direction * lipschitz_t), 1)
        f_next, grad_next = f_grad(x + step_size_t * update_direction)
        if (f_next - f_t) / step_size_t > - rho * certificate / 2:
          # sufficient decrease not met, increase Lipchitz constant
          lipschitz_t *= 2
          continue
        if (f_next - f_t) / step_size_t <= (rho / 2 - 1) * certificate:
          # there's sufficient decrease but the quadratic approximation is not
          # good. We can decrease the Lipschitz / increase the step-size
          lipschitz_t /= 1.5
          continue
        break
      else:
        warnings.warn(
            "Exhausted line search iterations in minimize_frank_wolfe",
            RuntimeWarning)
    elif step_size == "adaptive4":
      sigma = 0.7
      rho = 0.5
      for i in range(max_iter):
        step_size_t = min(certificate / (norm_update_direction * lipschitz_t), 1)
        f_next, grad_next = f_grad(x + step_size_t * update_direction)
        if (f_next - f_t) / step_size_t < - sigma * certificate / 2:
          # we can decrease the Lipschitz / increase the step-siPze
          lipschitz_t /= 1.5
          continue
        if (f_next - f_t) / step_size_t > - rho * certificate / 2:
          lipschitz_t *= 2.
          continue
        break
    elif step_size == "adaptive5":
      sigma = 0.9
      rho = 0.4
      eps = .3
      K = 2 * lipschitz_t * norm_update_direction / certificate
      M = max((K + sigma) / (K + rho), 1.)
      tau = M * (1 + eps)
      eta = (1 - eps) / M

#       print("Let's check multipliers")
#       print(tau)
#       print(eta)

      for i in range(max_iter):
        step_size_t = min(certificate / (norm_update_direction * lipschitz_t), 1)
        f_next, grad_next = f_grad(x + step_size_t * update_direction)
        if (f_next - f_t) / step_size_t < - sigma * certificate / 2:
          # we can decrease the Lipschitz / increase the step-size
          lipschitz_t *= eta
          continue
        if (f_next - f_t) / step_size_t > - rho * certificate / 2:
          lipschitz_t *= tau
          continue
        break

      else:
        warnings.warn(
            "Exhausted line search iterations in minimize_frank_wolfe", RuntimeWarning)
    elif step_size == "DR":
      # .. Demyanov-Rubinov step-size ..
      if lipschitz is None:
        raise ValueError("lipschitz needs to be specified with step_size=\"DR\"")
      step_size_t = min(certificate / (norm_update_direction * lipschitz_t), 1)
      f_next, grad_next = f_grad(x + step_size_t * update_direction)
    elif step_size is None:
      # .. without knowledge of the Lipschitz constant ..
      # .. we take the oblivious 2/(k+2) step-size ..
      step_size_t = 2. / (it+2)
      f_next, grad_next = f_grad(x + step_size_t * update_direction)
    else:
      raise ValueError("Invalid option step_size=%s" % step_size)
    if callback is not None:
      callback(locals())
    x += step_size_t * update_direction
    pbar.set_postfix(tol=certificate, iter=it, L_t=lipschitz_t)

    old_f_t, old_grad = f_t, grad
    f_t, grad = f_next, grad_next
  if callback is not None:
    callback(locals())
  pbar.close()
  return optimize.OptimizeResult(x=x, nit=it, certificate=certificate)


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


def minimize_pairwise_frank_wolfe(f_grad,
                                  x0,
                                  lmo,
                                  lmo_active,
                                  step_size=None,
                                  lipschitz=None,
                                  max_iter=200,
                                  tol=1e-12,
                                  callback=None,
                                  verbose=0):
  """Pairwise FW on the L1 ball.

.. warning::
    This feature is experimental, API is likely to change.

    """
  x0 = np.asanyarray(x0)
  n_features = x0.size

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

    fw_gap = grad[idx_oracle_away % n_features] * mag_away - \
          grad[idx_oracle % n_features] * mag_oracle
    if fw_gap <= tol:
      break

    norm_update_direction = 2 * (alpha**2)
    if backtracking:
      # because of the specific form of the update
      # we can achieve some extra efficiency this way
      for i in range(100):
        x_next = x.copy()
        step_size = min(fw_gap / (norm_update_direction * lipschitz_t), gamma_max)

        x_next[idx_oracle % n_features] += step_size * mag_oracle
        x_next[idx_oracle_away % n_features] -= step_size * mag_away
        f_next, grad_next = f_grad(x_next)
        if step_size < 1e-7:
          break
        elif f_next - f_t <= -fw_gap * step_size + 0.5 * (step_size**
                                                       2) * lipschitz_t * norm_update_direction:
          if i == 0:
            lipschitz_t *= 0.999
          break
        else:
          lipschitz_t *= 2
      # import pdb; pdb.set_trace()
    else:
      x_next = x.copy()
      step_size = min(fw_gap / (norm_update_direction * lipschitz_t), gamma_max)
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
        tol=fw_gap,
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
  return optimize.OptimizeResult(x=x, nit=it, certificate=fw_gap)
