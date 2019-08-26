

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
.. image:: https://storage.googleapis.com/copt/doc_status.svg
   :target: https://storage.googleapis.com/copt/index.html
.. image:: https://storage.googleapis.com/copt/pylint.svg
   :target: https://storage.googleapis.com/copt/pylint.txt

copt is a library for mathematical optimization written in pure Python.


Philosophy
----------

   * Modular, general-purpose optimization library.
   * API that follows whenever possible that of scipy.optimize.
   * State of the art performance, with emphasis on large-scale optimization.
   * Few dependencies, pure Python library for easy deployment.


Optimization algorithms
-----------------------
.. autosummary::

copt contains implementations of different optimization methods. These are categorized as:

.. admonition:: Proximal-gradient

  These are methods that combine the gradient of a smooth term with the proximal operator of a potentially non-smooth term.
  They can be used to solve problems involving one or several non-smooth terms :ref:`read more ...<proximal_gradient>`

.. admonition:: Frank-Wolfe

    Frank-Wolfe (also known as conditional gradient and projection-free methods) are a family of methods XXXX


.. admonition:: Stochastic Methods


  * Variance-reduced stochastic methods: :meth:`SAGA <copt.minimize_saga>`, :meth:`SVRG <copt.minimize_svrg>`, :meth:`variance-reduced three operator splitting <copt.minimize_vrtos>`


Getting started
---------------

If you already have a working installation of numpy and scipy,
the easiest way to install copt is using ``pip`` ::

    pip install -U copt


Alternatively, you can install the latest development from github with the command::

    pip install git+https://github.com/openopt/copt.git


.. warning::

    where to go from here?

.. toctree::
    :maxdepth: 2
    :hidden:

    proximal_gradient.rst
    frank_wolfe.rst
    incremental.rst
    loss_functions.rst
    auto_examples/index.rst
    datasets.rst
    utils.rst
    citing.rst
