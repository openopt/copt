
Loss and regularization functions
=================================

These are some convenience functions that implement wior convenience, the following are already defined and ready to use.

Smooth loss functions:


.. autosummary::
   :toctree: generated/

    copt.loss.LogLoss
    copt.loss.SquareLoss
    copt.loss.HuberLoss

Non-smooth terms accessed through their proximal operator

.. autosummary::
   :toctree: generated/

    copt.penalty.L1Norm
    copt.penalty.L1Ball
    copt.penalty.GroupL1
    copt.penalty.TraceNorm
    copt.penalty.TraceBall
    copt.penalty.TotalVariation2D
