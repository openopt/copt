"""COPT: composite optimization in Python."""
__version__ = '0.7.2'

from . import datasets
from . import tv_prox
from . import utils
from .frank_wolfe import minimize_frank_wolfe
from .frank_wolfe import minimize_pairwise_frank_wolfe
from .proximal_gradient import minimize_proximal_gradient
from .randomized import minimize_saga
from .randomized import minimize_svrg
from .randomized import minimize_vrtos
from .splitting import minimize_primal_dual
from .splitting import minimize_three_split
