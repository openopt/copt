

Welcome to copt!
================

.. image:: https://travis-ci.org/openopt/copt.svg?branch=master
   :target: https://travis-ci.org/openopt/copt
.. image:: https://coveralls.io/repos/github/openopt/copt/badge.svg?branch=master
   :target: https://coveralls.io/github/openopt/copt?branch=master
.. image:: https://img.shields.io/github/license/openopt/copt
   :alt: Apache-2 license
   :target: https://github.com/openopt/copt/blob/master/LICENSE
.. image:: https://badge.fury.io/py/copt.svg
   :target: https://badge.fury.io/py/copt
.. image:: https://zenodo.org/badge/46262908.svg
   :target: citing.html
.. image:: https://storage.googleapis.com/copt/pylint.svg


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

 * Proximal gradient: :meth:`proximal gradient descent <copt.minimize_proximal_gradient>`

 * Proximal splitting: :meth:`three operator splitting <copt.minimize_three_split>`, :meth:`primal-dual hybrid gradient <copt.minimize_primal_dual>`

 * Frank-Wolfe: :meth:`Frank-Wolfe <copt.minimize_frank_wolfe>`, :meth:`Pairwise Frank-Wolfe <copt.minimize_pairwise_frank_wolfe>`

 * Variance-reduced stochastic methods: :meth:`SAGA <copt.minimize_saga>`, :meth:`SVRG <copt.minimize_svrg>`, :meth:`variance-reduced three operator splitting <copt.minimize_vrtos>`


Getting started
---------------

If you already have a working installation of numpy and scipy,
the easiest way to install copt is using ``pip`` ::

    pip install -U copt


Alternatively, you can install the latest development from github with the command::

    pip install git+https://github.com/openopt/copt.git




.. toctree::
    :maxdepth: 2
    :hidden:

    proximal_gradient.rst
    proximal_splitting.rst
    frank_wolfe.rst
    incremental.rst
    loss_functions.rst
    auto_examples/index.rst
    datasets.rst
    utils.rst
    citing.rst
