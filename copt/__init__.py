"""COPT: composite optimization in Python."""
__version__ = "0.8.1"  # if you modify this, change it also in setup.py

from . import datasets
from . import tv_prox
from . import utils
from .frank_wolfe import minimize_frank_wolfe
from .proximal_gradient import minimize_proximal_gradient
from .randomized import minimize_saga
from .randomized import minimize_svrg
from .randomized import minimize_vrtos
from .splitting import minimize_primal_dual
from .splitting import minimize_three_split
