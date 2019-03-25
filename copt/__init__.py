"""COPT: composite optimization in Python."""
__version__ = '0.6.0'

from .proxgrad import minimize_proxgrad
from .splitting import minimize_three_split, minimize_primal_dual
from .randomized import minimize_saga, minimize_SVRG, minimize_vrtos
from .frank_wolfe import minimize_fw, minimize_pfw_l1
from . import datasets
from . import tv_prox
from . import utils
