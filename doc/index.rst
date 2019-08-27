

Welcome to copt!
================

.. image:: https://travis-ci.org/openopt/copt.svg?branch=master
   :target: https://travis-ci.org/openopt/copt
.. image:: https://storage.googleapis.com/copt/doc_status.svg
   :target: https://storage.googleapis.com/copt/index.html
.. image:: https://coveralls.io/repos/github/openopt/copt/badge.svg?branch=master
   :target: https://coveralls.io/github/openopt/copt?branch=master
.. image:: https://storage.googleapis.com/copt/pylint.svg
   :target: https://storage.googleapis.com/copt/pylint.txt
.. image:: https://zenodo.org/badge/46262908.svg
   :target: citing.html

copt is a library for mathematical optimization written in pure Python.


Philosophy
----------

  * State of the art implementation of classical optimization algorithms.
  * API that follows whenever possible that of scipy.optimize.
  * State of the art performance, with emphasis on large-scale optimization.
  * Few dependencies, pure Python library for easy deployment.


Contents
-----------------------

The methods implements in copt can be categorized as:

.. admonition:: Proximal-gradient

  These are methods that combine the gradient of a smooth term with the proximal operator of a potentially non-smooth term.
  They can be used to solve problems involving one or several non-smooth terms. :ref:`Read more ...<proximal_gradient>`

.. admonition:: Frank-Wolfe

    Frank-Wolfe, also known as conditional gradient, are a family of methods to solve constrained optimization problems. Contrary to proximal-gradient methods, they don't require access to the projection onto the constraint set. :ref:`Read more ...<frank_wolfe>`


.. admonition:: Stochastic Methods

  Methods that can solve optimization problems with access only to a noisy evaluation of the objective.
  :ref:`Read more ...<stochastic_methods>`.


Installation
------------

If you already have a working installation of numpy and scipy,
the easiest way to install copt is using ``pip`` ::

    pip install -U copt


Alternatively, you can install the latest development from github with the command::

    pip install git+https://github.com/openopt/copt.git



Where to go from here?
----------------------

To know more about copt, check out our :ref:`example gallery <sphx_glr_auto_examples>` or browse through the module reference using the left navigation bar.


.. toctree::
    :maxdepth: 2
    :hidden:

    proximal_gradient
    frank_wolfe
    stochastic
    loss_functions
    auto_examples/index
    utils
    citing
