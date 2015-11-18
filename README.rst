gdprox, proximal gradient-descent algorithms
============================================

Implements the proximal gradient-descent algorithm for composite objective functions, i.e. functions of the form :code:`f(x) + g(x)`, where f is a smooth function and g is a possibly non-smooth function for which the proximal operator is known. 

The main function in this package is :code:`gdprox.fmin_cgprox`. This function follows a similar interface than the functions in :code:`scipy.optimize`. The definition of this function is:


.. code-block:: python

	def fmin_cgprox(f, fprime, g_prox, x0, rtol=1e-6,
	                maxiter=1000, verbose=0, default_step_size=1.):
	    """
	    proximal gradient-descent solver for optimization problems of the form

	                       minimize_x f(x) + g(x)

	    where f is a smooth function and g is a (possibly non-smooth)
	    function for which the proximal operator is known.

	    Parameters
	    ----------
	    f : callable
	        f(x) returns the value of f at x.

	    f_prime : callable
	        f_prime(x) returns the gradient of f.

	    g_prox : callable of the form g_prox(x, alpha)
	        g_prox(x, alpha) returns the proximal operator of g at x
	        with parameter alpha.

	    x0 : array-like
	        Initial guess

	    maxiter : int
	        Maximum number of iterations.

	    verbose : int
	        Verbosity level, from 0 (no output) to 2 (output on each iteration)

	    default_step_size : float
	        Starting value for the line-search procedure.

	    Returns
	    -------
	    res : OptimizeResult
	        The optimization result represented as a
	        ``scipy.optimize.OptimizeResult`` object. Important attributes are:
	        ``x`` the solution array, ``success`` a Boolean flag indicating if
	        the optimizer exited successfully and ``message`` which describes
	        the cause of the termination. See `scipy.optimize.OptimizeResult`
	        for a description of other attributes.
	    """


