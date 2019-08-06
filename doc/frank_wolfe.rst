.. _frank_wolfe:

Frank-Wolfe and other projection-free algorithms
================================================


Frank-Wolfe
-----------

The Frank-Wolfe (FW) or conditional gradient algorithm [1]_, [2]_, [3]_ is a method for constrained optimization. It can solve problems of the form  f

.. math::
      \argmin_{\bs{x} \in \mathcal{D}} f(\bs{x})

where :math:`f`. is differentiable and the domain :math:`\mathcal{D}` is a convex and compart set.

Contrary to other constrained optimization algorithms like projected gradient descent, the Frank-Wolfe algorithm does not require access to a projection, hence why it is sometimes referred to as a projection-free algorithm. It instead relies on a routine that solves a linear problem over the domain. We refer to this routine as a linear minimization oracle (keyword argument lmo).

The Frank-Wolfe algorithm is implemented in this library in the method :meth:`copt.minimize_frank_wolfe`. As most other methods it takes as argument an objective function to minimize, but unlike most other methods, it also requires to 

At each iteration, the Frank-Wolfe algorithm selects the vertex :math:`\boldsymbol{s}_t` of the domain that correlates the most with the negative gradient (thanks to the linear minimization oracle) and then constructs the next iterate :math:`\boldsymbol{x}_{t+1}` as a convex combination of the current iterate :math:`\boldsymbol{x}_{t}` and the newly acquired vertex :math:`\boldsymbol{s}_t`:

.. math::
      \boldsymbol{x}_{t+1} = (1 - \gamma)\boldsymbol{x}_t + \gamma \boldsymbol{s}_t



.. .. image:: http://fa.bianp.net/images/2018/FW_iterates.png
..    :alt: FW iterates
..    :align: center




.. autosummary::
   :toctree: generated/

    copt.minimize_frank_wolfe


.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_frank_wolfe_plot_fw_stepsize.py`
   * :ref:`sphx_glr_auto_examples_frank_wolfe_plot_fw_vertex_overlap.py`



.. topic:: References:

    .. [1] Jaggi, Martin. `"Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization." <http://proceedings.mlr.press/v28/jaggi13-supp.pdf>`_ ICML 2013.

    .. [2] Pedregosa, Fabian `"Notes on the Frank-Wolfe Algorithm" <http://fa.bianp.net/blog/2018/notes-on-the-frank-wolfe-algorithm-part-i/>`_, 2018

    .. [3] Pedregosa, Fabian, Armin Askari, Geoffrey Negiar, and Martin Jaggi. `"Step-Size Adaptivity in Projection-Free Optimization." <https://arxiv.org/pdf/1806.05123.pdf>`_ arXiv:1806.05123 (2018).


Pairwise Frank-Wolfe
--------------------

As the Frank-Wolfe algorithm, the Pairwise Frank-Wolfe [4]_ solves problems of the form 

.. math::
      \argmin_{\bs{x} \in \mathcal{D}} f(\bs{x})

where :math:`f`. is differentiable and the domain :math:`\mathcal{D}` is a convex and compart set.

Although the algorithm is more broadly applicable, this library's implementation, :meth:`copt.minimize_pairwise_frank_wolfe_l1`, assumes that the domain :math:`\mathcal{D}` is the :math:`\ell_1` ball, that is, :math:`\mathcal{D} = \{x : \sum_i |x| \leq \alpha\}`, where :math:`\alpha` is a user-defined parameter.


.. autosummary::
   :toctree: generated/

    copt.minimize_pairwise_frank_wolfe_l1


.. topic:: References:

  .. [4] Lacoste-Julien, Simon, and Martin Jaggi. "On the global linear convergence of Frank-Wolfe optimization variants." Advances in Neural Information Processing Systems. 2015.