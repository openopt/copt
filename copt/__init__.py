__version__ = '0.3.0-dev'

from .gradient import minimize_PGD, minimize_DavisYin, minimize_APGD
from .primal_dual import fmin_CondatVu
from .randomized import minimizelp_SAGA, minimize_BCD
from . import datasets
from . import tv_prox
from . import utils
