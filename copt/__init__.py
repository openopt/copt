__version__ = '0.4.0'

from .gradient import minimize_PGD, minimize_TOS, minimize_APGD, minimize_PDHG
from .randomized import minimize_SAGALP_L1, minimize_BCD
from . import datasets
from . import tv_prox
from . import utils
from . import frank_wolfe
