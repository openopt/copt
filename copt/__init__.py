__version__ = '0.1.dev0'

from .gradient_descent import fmin_PGD, fmin_DavisYin, fmin_APGD
from .primal_dual import fmin_CondatVu
from .stochastic import fmin_SAGA, fmin_PSSAGA
from .utils import LogisticLoss, L1Norm, TotalVariation2D
from . import datasets
from . import tv_prox
