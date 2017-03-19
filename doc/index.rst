

Welcome to copt!
================

copt is a library for mathematical optimization written in pure Python.

Philosophy
----------

   * Modular, general-purpose optimization library.
   * State of the art performance, with particular emphasis on large-scale problems.
   * Few dependencies, pure Python library for easy deployment.

.. warning::
    This is a work in progress, things will break.


Optimization algorithms
-----------------------
.. autosummary::
   :toctree: generated/

    copt.minimize_SAGA
    copt.minimize_PGD
    copt.minimize_APGD
    copt.minimize_DavisYin
    copt.minimize_CondatVu


Supported loss functions
------------------------

It is possible to your own functions. However, for convenience, the following are already defined and ready to use.

Smooth loss functions:

.. autosummary::
   :toctree: generated/

    copt.LogisticLoss
    copt.SquaredLoss

Nonsmooth loss functions:

.. autosummary::
   :toctree: generated/

    copt.L1Norm
    copt.TotalVariation2D



.. toctree::
    :hidden:
    :glob:

    *
    auto_examples/index.rst