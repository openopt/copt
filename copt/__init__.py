"""COPT: composite optimization in Python."""
__version__ = '0.6.0'

from .proximal_gradient import minimize_proximal_gradient
from .splitting import minimize_three_split
from .splitting import minimize_primal_dual
from .randomized import minimize_saga, minimize_svrg, minimize_vrtos
from .frank_wolfe import minimize_frank_wolfe, minimize_pairwise_frank_wolfe_l1
from . import datasets
from . import tv_prox
from . import utils
