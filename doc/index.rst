

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

    copt.fmin_PGD
    copt.fmin_DavisYin
    copt.fmin_CondatVu
    copt.fmin_SAGA
    copt.fmin_PSSAGA


Proximal operators
------------------
.. autosummary::
   :toctree: generated/

    copt.prox.prox_tv1d



.. toctree::
    :hidden:
    :glob:

    *
    auto_examples/index.rst