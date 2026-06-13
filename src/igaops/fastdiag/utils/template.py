from typing import List, Dict, Tuple, Sequence, Any
from abc import ABC, abstractmethod

from igaops.common import ParametricDirection
from igaops.quadrature import StandardGauss


class Template(ABC):
    """
    This class implements the fast-diagonalization preconditioner for a single patch.
    """

    @property
    @abstractmethod
    def nonzeros_by_dir(self) -> Dict[ParametricDirection, int]:
        raise NotImplementedError("To implement in children")

    @property
    @abstractmethod
    def indices_by_dir(self) -> Dict[Tuple[int, ParametricDirection], List[int]]:
        raise NotImplementedError("To implement in children")

    @property
    @abstractmethod
    def nbDoFsPerNode(self) -> int:
        raise NotImplementedError("To implement in children")

    @staticmethod
    def rewrite_quadrature(
        quadrule_list: Sequence[Any],
    ) -> List[StandardGauss]:
        def verify(q: Any) -> StandardGauss:
            if isinstance(q, StandardGauss) and q.quadrature_type == "legendre":
                return q

            q = StandardGauss(
                degree=q.degree,
                knotvector=q.knotvector,
                quadtype="legendre",
            )
            return q

        if not isinstance(quadrule_list, (list, tuple)):
            raise ValueError("quadrule_list should be iterable")
        return [verify(q) for q in quadrule_list]
