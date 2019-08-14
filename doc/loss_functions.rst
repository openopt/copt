
Loss and regularization functions
=================================

It is possible to your own functions. However, for convenience, the following are already defined and ready to use.

Smooth loss functions:


.. autosummary::
   :toctree: generated/

    copt.utils.LogLoss
    copt.utils.SquareLoss
    copt.utils.HuberLoss

Non-smooth terms accessed through their proximal operator

.. autosummary::
   :toctree: generated/

    copt.utils.L1Norm
    copt.utils.L1Ball
    copt.utils.GroupL1
    copt.utils.TraceNorm
    copt.utils.TraceBall
    copt.utils.TotalVariation2D
