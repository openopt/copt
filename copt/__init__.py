__version__ = '0.1.dev0'

from .gradient_descent import fmin_PGD, fmin_DavisYin
from .prox import prox_tv2d, prox_tv1d
from .primal_dual import fmin_CondatVu
from .stochastic import fmin_SAGA
from .stochastic import fmin_PSSAGA