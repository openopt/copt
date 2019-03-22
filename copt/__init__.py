__version__ = '0.6.0'

from .proxgrad import minimize_PGD, minimize_APGD
from .splitting import minimize_TOS, minimize_PDHG
from .randomized import minimize_saga, minimize_SVRG, minimize_vrtos
from .frank_wolfe import minimize_fw, minimize_pfw_l1
from . import datasets
from . import tv_prox
from . import utils
