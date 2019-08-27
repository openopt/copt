.. _proximal_gradient:

Gradient-based methods
======================


Proximal-Gradient
-----------------

The proximal-gradient method [BT2009]_, [N2013]_ is a method to solve problems of the form

.. math::
      \argmin_{\bs{x} \in \mathbb{R}^d} f(\bs{x}) + g(\bs{x})


where $f$ is a differentiable function for which we have access to its gradient and $g$ is a potentially non-smooth function for which we have access to its proximal operator.

.. autosummary::
   :toctree: generated/

    copt.minimize_proximal_gradient


.. admonition:: Examples

  * :ref:`sphx_glr_auto_examples_plot_group_lasso.py`


.. topic:: References

  .. [BT2009] Beck, Amir, and Marc Teboulle. `"Gradient-based algorithms with applications to signal recovery." <https://pdfs.semanticscholar.org/e7a7/5a379a515197e058102d985cd80f4f047c04.pdf>`_ Convex optimization in signal processing and communications (2009)

  .. [N2013] Nesterov, Yu. `"Gradient methods for minimizing composite functions." <https://doi.org/10.1007/s10107-012-0629-5>`_ Mathematical Programming 140.1 (2013): 125-161.


Primal-dual hybrid gradient
---------------------------

The primal-dual hybrid gradient method [C2013]_ [V2013]_ [CP2016]_ is a method to solve problems of the form

.. math::
      \argmin_{\bs{x} \in \mathbb{R}^d} f(\bs{x}) + g(\bs{x}) + h(\bs{A}\bs{x})

where $f$ is a differentiable function for which we have access to its gradient and $g$ and $h$ are potentially non-smooth functions for which we have access to their proximal operator.

.. autosummary::
  :toctree: generated/
  
  copt.minimize_primal_dual


.. admonition:: Examples

   * :ref:`sphx_glr_auto_examples_proximal_splitting_plot_tv_deblurring.py`


.. topic:: References

  .. [C2013] Condat, Laurent. "A primal–dual splitting method for convex optimization involving Lipschitzian, proximable and linear composite terms." Journal of Optimization Theory and Applications 158.2 (2013): 460-479.

  .. [V2013] Vũ, Bằng Công. "A splitting algorithm for dual monotone inclusions involving cocoercive operators." Advances in Computational Mathematics 38.3 (2013)

  .. [CP2016] Chambolle, Antonin, and Thomas Pock. "An introduction to continuous optimization for imaging." Acta Numerica 25 (2016) 


Three-operator splitting
------------------------

The three operator splitting [DY2017]_ [PG2018]_ is a method to solve problems of the form

.. math::
      \argmin_{\bs{x} \in \mathbb{R}^d} f(\bs{x}) + g(\bs{x}) + h(\bs{x})

where $f$ is a differentiable function for which we have access to its gradient and $g$ and $h$ are potentially non-smooth functions for which we have access to their proximal operator.

.. autosummary::
  :toctree: generated/

  copt.minimize_three_split



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
