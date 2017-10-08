__version__ = '0.3.0-dev'

from .gradient import minimize_PGD, minimize_DavisYin, minimize_APGD
from .primal_dual import fmin_CondatVu
from .randomized import minimize_SAGA, minimize_BCD
from .utils import LogisticLoss, SquaredLoss, L1Norm, TotalVariation2D
from . import datasets
from . import tv_prox
