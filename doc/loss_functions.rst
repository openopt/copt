
Loss, constraints and regularizers
==================================

These are some convenience functions that implement common losses, constraints and regularizers.

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
    copt.penalty.GroupL1
    copt.penalty.TraceNorm
    copt.penalty.FusedLasso
    copt.penalty.TotalVariation2D


Constraints can be incorporated in a similar way through 


.. autosummary::
   :toctree: generated/

    copt.contraint.L1Ball
    copt.contraint.TraceBall
