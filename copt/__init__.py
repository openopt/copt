__version__ = '0.0.dev0'

from .gradient_descent import two_prox_grad
from .total_variation import prox_tv2d, prox_tv1d
from .proximal_splitting import three_DY, three_CV
from .stochastic import two_SAGA