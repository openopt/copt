.. _frank_wolfe:

Frank-Wolfe and other projection-free algorithms
================================================


Frank-Wolfe
-----------

The Frank-Wolfe (FW) or conditional gradient algorithm [1]_, [2]_ a method for constrained optimization and has seen an impressive revival in recent years due to its low memory requirement and projection-free iterations. It can solve problems of the form  

.. math::
      \argmin_{\bs{x} \in \mathcal{D}} f(\bs{x})

where :math:`f`. is differentiable and the domain :math:`\mathcal{D}` is a convex and compart set.


Contrary to other constrained optimization algorithms like projected gradient descent, the Frank-Wolfe algorithm does not require access to a projection, hence why it is sometimes referred to as a projection-free algorithm. It instead relies on a routine that solves a linear problem over the domain. We refer to this routine as a linear minimization oracle (keyword argument lmo).

.. autosummary::
   :toctree: generated/

    copt.minimize_frank_wolfe


.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_plot_fw_stepsize.py`
   * :ref:`sphx_glr_auto_examples_plot_fw_vertex_overlap.py`



.. topic:: References:

    .. [1] Jaggi, Martin. `"Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization." <http://proceedings.mlr.press/v28/jaggi13-supp.pdf>`_ ICML 2013.

    .. [2] Pedregosa, Fabian `"Notes on the Frank-Wolfe Algorithm" <http://fa.bianp.net/blog/2018/notes-on-the-frank-wolfe-algorithm-part-i/>`_, 2018

    .. [3] Pedregosa, Fabian, Armin Askari, Geoffrey Negiar, and Martin Jaggi. `"Step-Size Adaptivity in Projection-Free Optimization." <https://arxiv.org/pdf/1806.05123.pdf>`_ arXiv:1806.05123 (2018).


Pairwise Frank-Wolfe
--------------------

.. autosummary::
   :toctree: generated/

    copt.minimize_pfw_l1


