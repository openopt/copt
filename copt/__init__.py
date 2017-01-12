__version__ = '0.1.dev0'

from .gradient_descent import fmin_ProxGrad
from .prox import prox_tv2d, prox_tv1d
from .three_split import fmin_DavisYin, fmin_CondatVu
from .stochastic import fmin_SAGA, fmin_PSSAGA