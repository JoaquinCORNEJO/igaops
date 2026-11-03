from .preconditioner import SingleFastDiagonalization as SingleFD
from .multi_preconditioner import MultiFastDiagonalization as MultiFD

__all__ = ["SingleFD", "MultiFD"]
