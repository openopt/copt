.. _proximal_gradient:

Solvers
=======


Proximal-Gradient
-----------------

.. autosummary::
   :toctree: generated/

    copt.minimize_proximal_gradient

The proximal-gradient method [BT2009]_, [N2013]_ is a method to solve problems of the form

.. math::
      \argmin_{\bs{x} \in \mathbb{R}^d} f(\bs{x}) + g(\bs{x})


where $f$ is a differentiable function for which we have access to its gradient and $g$ is a potentially non-smooth function for which we have access to its proximal operator.


.. admonition:: Examples

  * :ref:`sphx_glr_auto_examples_plot_group_lasso.py`


.. topic:: References

  .. [BT2009] Beck, Amir, and Marc Teboulle. `"Gradient-based algorithms with applications to signal recovery." <https://pdfs.semanticscholar.org/e7a7/5a379a515197e058102d985cd80f4f047c04.pdf>`_ Convex optimization in signal processing and communications (2009)

  .. [N2013] Nesterov, Yu. `"Gradient methods for minimizing composite functions." <https://doi.org/10.1007/s10107-012-0629-5>`_ Mathematical Programming 140.1 (2013): 125-161.


Primal-dual hybrid gradient
---------------------------

.. autosummary::
  :toctree: generated/
  
  copt.minimize_primal_dual


The primal-dual hybrid gradient method [C2013]_ [V2013]_ [CP2016]_ is a method to solve problems of the form

.. math::
      \argmin_{\bs{x} \in \mathbb{R}^d} f(\bs{x}) + g(\bs{x}) + h(\bs{A}\bs{x})

where $f$ is a differentiable function for which we have access to its gradient and $g$ and $h$ are potentially non-smooth functions for which we have access to their proximal operator.



.. admonition:: Examples

   * :ref:`sphx_glr_auto_examples_proximal_splitting_plot_tv_deblurring.py`


.. topic:: References

  .. [C2013] Condat, Laurent. "A primal–dual splitting method for convex optimization involving Lipschitzian, proximable and linear composite terms." Journal of Optimization Theory and Applications 158.2 (2013): 460-479.

  .. [V2013] Vũ, Bằng Công. "A splitting algorithm for dual monotone inclusions involving cocoercive operators." Advances in Computational Mathematics 38.3 (2013)

  .. [CP2016] Chambolle, Antonin, and Thomas Pock. "An introduction to continuous optimization for imaging." Acta Numerica 25 (2016) 


Three-operator splitting
------------------------


.. autosummary::
  :toctree: generated/

  copt.minimize_three_split


The three operator splitting [DY2017]_ [PG2018]_ is a method to solve problems of the form

.. math::
      \argmin_{\bs{x} \in \mathbb{R}^d} f(\bs{x}) + g(\bs{x}) + h(\bs{x})

where $f$ is a differentiable function for which we have access to its gradient and $g$ and $h$ are potentially non-smooth functions for which we have access to their proximal operator.


.. admonition:: Examples

   * :ref:`sphx_glr_auto_examples_proximal_splitting_plot_sparse_nuclear_norm.py`
   * :ref:`sphx_glr_auto_examples_proximal_splitting_plot_tv_deblurring.py`
   * :ref:`sphx_glr_auto_examples_proximal_splitting_plot_overlapping_group_lasso.py`


.. topic:: References

  .. [DY2017] Davis, Damek, and Wotao Yin. `"A three-operator splitting scheme and
    its optimization applications."
    <https://doi.org/10.1007/s11228-017-0421-z>`_ Set-Valued and Variational
    Analysis, 2017.

  .. [PG2018] Pedregosa, Fabian, and Gauthier Gidel. `"Adaptive Three Operator
    Splitting." <https://arxiv.org/abs/1804.02339>`_ Proceedings of the 35th
    International Conference on Machine Learning, 2018.


.. _frank_wolfe:

Frank-Wolfe
-----------

.. autosummary::
  :toctree: generated/

    copt.minimize_frank_wolfe


The Frank-Wolfe (FW) or conditional gradient algorithm [J2003]_, [P2018]_, [PANJ2018]_ is a method for constrained optimization. It can solve problems of the form

.. math::
      \argmin_{\bs{x} \in \mathcal{D}} f(\bs{x})

where :math:`f` is a differentiable function for which we have access to its gradient and :math:`\mathcal{D}` is a compact set for which we have access to its linear minimization oracle (lmo). This is a routine that given a vector :math:`\bs{u}` returns a solution to

.. math::
    \argmin_{\bs{x} \in D}\, \langle\bs{u}, \bs{x}\rangle~.


Contrary to other constrained optimization algorithms like projected gradient descent, the Frank-Wolfe algorithm does not require access to a projection, hence why it is sometimes referred to as a projection-free algorithm. It instead relies exclusively on the linear minimization oracle described above.


.. TODO describe the LMO API in more detail


The Frank-Wolfe algorithm is implemented in this library in the method :meth:`copt.minimize_frank_wolfe`. As most other methods it takes as argument an objective function to minimize, but unlike most other methods, it requires access to a *linear minimization oracle*, which is a routine that for a given $d$-dimensional vector :math:`\bs{u}` solves the linear problems  :math:`\argmin_{\bs{z} \in D}\, \langle \bs{u}, \bs{z}\rangle`.


At each iteration, the Frank-Wolfe algorithm uses the linear minimization oracle to identify the vertex :math:`\bs{s}_t` that correlates most with the negative gradient. Then next iterate :math:`\boldsymbol{x}^+` is constructed as a convex combination of the current iterate :math:`\boldsymbol{x}` and the newly acquired vertex :math:`\boldsymbol{s}`:


.. math::
      \boldsymbol{x}^+ = (1 - \gamma)\boldsymbol{x} + \gamma \boldsymbol{s}



The step-size :math:`\gamma` can be chosen by different strategies:

  * **Inexact line-search**. This is the default option and corresponds to the keyword argument :code:`step_size="adaptive"` This is typically the fastest and simplest method, if unsure, use this option.

  * **Demyanov-Rubinov step-size**. This is a step-size of the form
    
    .. math::
        \gamma = \langle \nabla f(\bs{x}), \bs{s} - \bs{x}\rangle / (L \|\bs{s} - \bs{x}\|^2)~.



    This step-size typically performs well but has the drawback that it requires knowledge of the Lipschitz constant of :math:`\nabla f`. This step-size can be used with the keyword argument :code:`step_size="DR"`. In this case the Lipschitz
    constant :math:`L` needs to be specified through the keyword argument :code:`lipschitz`. For example, if the lipschitz constant is 0.1, then the signature should include :code:`step_size="DR", lipschitz=0.1`.


  * **Oblivious step-size**. This is the very simple step-size of the form
  
    .. math::
      \gamma = \frac{2}{t+2}~,
    
    where :math:`t` is the number of iterations. This step-size is oblivious since it doesn't use any previous information of the objective. It typically performs worst than the alternatives, but is simple to implement and can be competitive in the case in the case of noisy objectives.


Below is an illustration of the iterates generated by the Frank-Wolfe algorithkm on a toy 2-dimensional problem, in which the triangle is the domain  :math:`\mathcal{D}` and the level curves represent values of the objective function  :math:`f`.

.. image:: http://fa.bianp.net/images/2018/FW_iterates.png
  :alt: FW iterates
  :align: center



.. admonition:: Examples

  * :ref:`sphx_glr_auto_examples_frank_wolfe_plot_sparse_benchmark.py`
  * :ref:`sphx_glr_auto_examples_frank_wolfe_plot_vertex_overlap.py`
  * :ref:`sphx_glr_auto_examples_frank_wolfe_plot_sparse_benchmark_pairwise.py`



.. topic:: References:

  .. [J2003] Jaggi, Martin. `"Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization." <http://proceedings.mlr.press/v28/jaggi13-supp.pdf>`_ ICML 2013.

  .. [P2018] Pedregosa, Fabian `"Notes on the Frank-Wolfe Algorithm" <http://fa.bianp.net/blog/2018/notes-on-the-frank-wolfe-algorithm-part-i/>`_, 2018

  .. [PANJ2018] Pedregosa, Fabian, Armin Askari, Geoffrey Negiar, and Martin Jaggi. `"Step-Size Adaptivity in Projection-Free Optimization." <https://arxiv.org/pdf/1806.05123.pdf>`_ arXiv:1806.05123 (2018).


  .. [LJ2015] Lacoste-Julien, Simon, and Martin Jaggi. `"On the global linear convergence of Frank-Wolfe optimization variants." <https://arxiv.org/pdf/1511.05932.pdf>`_ Advances in Neural Information Processing Systems. 2015.




.. _stochastic_methods:

Stochastic methods
------------------

.. autosummary::
   :toctree: generated/

    copt.minimize_saga
    copt.minimize_svrg
    copt.minimize_vrtos


.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_plot_saga_vs_svrg.py`
