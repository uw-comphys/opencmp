# Superclass
from .base_model import Model

# Implemented models
from .ins import INS
from .poisson import Poisson
from .stokes import Stokes

# Helper functions
from .misc import get_model_class
