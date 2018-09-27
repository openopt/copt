

Welcome to copt!
================

copt is a library for mathematical optimization written in pure Python.

Philosophy
----------

   * Modular, general-purpose optimization library.
   * API similar to that of scipy.optimize.
   * State of the art performance, with emphasis on large-scale optimization.
   * Few dependencies, pure Python library for easy deployment.


Optimization algorithms
-----------------------
.. autosummary::

C-OPT contains implementations of different optimization methods. These are categorized as:

 * :ref:`gradient_methods`: :meth:`proximal gradient descent <copt.minimize_PGD>` and its :meth:`accelerated variant <copt.minimize_APGD>`, :meth:`three operator splitting <copt.minimize_TOS>`
 * :ref:`incremental_methods`: :meth:`SAGA <copt.minimize_SAGA_L1>`, :meth:`SVRG <copt.minimize_SVRG_L1>`



.. toctree::
    :hidden:
    :glob:

    gradient.rst
    incremental.rst
    loss_functions.rst
    datasets.rst
    auto_examples/index.rst


.. warning::
    This library is a work in progress, expect some rough edges.
