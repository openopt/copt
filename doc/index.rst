

Welcome to copt!
================

copt is a library for mathematical optimization written in pure Python.

Philosophy
----------

   * Modular, general-purpose optimization library.
   * State of the art performance, with particular emphasis on large-scale problems.
   * Few dependencies, pure Python library for easy deployment.

.. warning::
    This library is a work in progress, expect some rough edges.


Optimization algorithms
-----------------------
.. autosummary::
   :toctree: generated/

    copt.minimize_SAGA
    copt.minimize_PGD
    copt.minimize_APGD
    copt.minimize_DavisYin


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
    copt.L1Ball
    copt.TotalVariation2D
    copt.TraceNorm

Datasets
--------

.. autosummary::
   :toctree: generated/

    copt.datasets.load_img1
    copt.datasets.load_rcv1
    copt.datasets.load_url
    copt.datasets.load_covtype

.. toctree::
    :hidden:
    :glob:

    *
    auto_examples/index.rst
