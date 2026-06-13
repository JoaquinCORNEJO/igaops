from typing import Sequence, Union, List, Optional
from abc import ABC, abstractmethod

from scipy import sparse as sp
import numpy as np

from igaops.quadrature import StandardGauss, WeightedQuadrature

QuadratureRule = Union[StandardGauss, WeightedQuadrature]


class Template(ABC):

    @staticmethod
    @abstractmethod
    def eval_hessian(
        basis_list: Sequence[List[sp.csr_array]],
        u_at_ctrlpts: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
        along_axis: bool = False,
    ) -> np.ndarray:
        """
        Evaluate the Hessian of a rank-1 tensor field u at quadrature points.

        Let say the 1-rank tensor field u is represented as u = sum_A N_A u_A,
        where N_A are NURBS (or B-spline) functions in the parametric space
        and u_A are the 'control points' of the field on the physical space.
        The Hessian is defined as H_ABC = d / d x_C (d u_A / d x_B).
        Here x_B and x_C are the components of the parametric space.

        Parameters
        ----------
        basis_list : List[List[array_like]]
            List of basis functions.
        u_at_ctrlpts : ndarray
            The 'control points' of the vector field u.
        nurbs_weights : ndarray, optional
            The weights of the B-spline basis to compute the NURBS basis.
            Defaults to an empty array.
        along_axis : bool
            If true uses matrix-free tensor product, otherwise it uses kronecker product
            along 1-axis (0-axis is fixed). Defaults to False.

        Returns
        -------
        ndarray
            The Hessian of the rank-1 field u evaluated at the points given
            in the list of quadrature rules.
        """
        raise NotImplementedError("To implement in children")

    @staticmethod
    @abstractmethod
    def eval_jacobien(
        basis_list: Sequence[List[sp.csr_array]],
        u_at_ctrlpts: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
        along_axis: bool = False,
    ) -> np.ndarray:
        """
        Evaluate the Jacobian of a rank-1 tensor field u at quadrature points.

        Let say the 1-rank tensor field u is represented as u = sum_A N_A u_A,
        where N_A are NURBS (or B-spline) functions in the parametric space
        and u_A are the 'control points' of the field on the physical space.
        The jacobien is defined as J_AB = d u_A / d x_B. Here x_B are the
        components of the parametric space.

        Parameters
        ----------
        basis_list : List[List[array_like]]
            List of basis functions.
        u_at_ctrlpts : ndarray
            The 'control points' of the vector field u.
        nurbs_weights : ndarray
            The weights of the B-spline basis to compute the NURBS basis.
            Defaults to an empty array.
        along_axis : bool
            If true uses matrix-free tensor product, otherwise it uses kronecker product
            along 1-axis (0-axis is fixed). Defaults to False.

        Returns
        -------
        ndarray
            The jacobien of the 1-rank field u evaluated the points given in
            the list of quadrature rules (by default quadrature points)
        """
        raise NotImplementedError("To implement in children")

    @staticmethod
    @abstractmethod
    def eval_interpolation(
        basis_list: Sequence[List[sp.csr_array]],
        u_at_ctrlpts: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
        along_axis: bool = False,
    ) -> np.ndarray:
        """
        Evaluate the interpolation of a rank-1 tensor field u at quadrature points.

        Let say the 1-rank tensor field u is represented as u = sum_A N_A u_A,
        where N_A are NURBS (or B-spline) functions in the parametric space
        and u_A are the 'control points' of the field on the physical space.
        The interpolation is just the projection of u on a set of given points.

        Parameters
        ----------
        basis_list : List[List[array_like]]
            List of basis functions.
        u_at_ctrlpts : ndarray
            The 'control points' of the vector field u.
        nurbs_weights : ndarray
            The weights of the B-spline basis to compute the NURBS basis.
            Defaults to an empty array.
        along_axis : bool
            If true uses matrix-free tensor product, otherwise it uses kronecker product
            along 1-axis (0-axis is fixed). Defaults to False.

        Returns
        -------
        ndarray
            The interpolation of the 1-rank field u at the points given in
            the list of quadrature rules (by default quadrature points)
        """
        raise NotImplementedError("To implement in children")

    @staticmethod
    @abstractmethod
    def assemble_scalar_u_v(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> sp.csr_array:
        """
        Assembles the matrix M (also called mass matrix)

        It results from
        'int N_A(x) c(x) N_B(x) dx' <- in the hypercube [0, 1]^d
        where N_A and N_B are basis functions in the same parametric space.

        Parameters
        ----------
        quadrule_list : List[QuadratureRule]
            List of quadrature rules to build the basis functions and to compute the integral.
        coefficients : ndarray
            Contains geometry and material properties. It is a 0-rank tensor (or scalar) field.
        nurbs_weights : ndarray
            The weights of the B-spline basis to compute the NURBS basis.

        Notes:
        ------
        This function is not optimize since it uses python loops. Avoid using it if possible.
        """
        raise NotImplementedError("To implement in children")

    @staticmethod
    @abstractmethod
    def assemble_scalar_gradu_gradv(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> sp.csr_array:
        """
        Assembles the matrix M (also called stiffness matrix)

        It results from
        'int grad(N_A(x)) [c(x) . grad(N_B(x))] dx' <- in the hypercube [0, 1]^d
        where N_A and N_B are basis functions in the same parametric space.

        Parameters
        ----------
        quadrule_list : List[QuadratureRule]
            List of quadrature rules to build the basis functions and to compute the integral.
        coefficients : ndarray
            Contains geometry and material properties. It is a 2-rank tensor field.
        nurbs_weights : ndarray
            The weights of the B-spline basis to compute the NURBS basis.

        Notes:
        ------
        This function is not optimize since it computes the kron product directly and
        don't use matrix-free algorithms. Avoid using it if possible.
        """
        raise NotImplementedError("To implement in children")

    @staticmethod
    @abstractmethod
    def assemble_scalar_u_force(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> np.ndarray:
        """
        Computes the force-like array F using matrix-free algorithms.

        Here the terms of F results from
        'int N_A(x) c(x) dx' <- in the hypercube [0, 1]^d
        where N_A are basis functions in the same parametric space.

        Parameters
        ----------
        quadrule_list : List[QuadratureRule]
            List of quadrature rules to build the basis functions and to compute the integral.
        coefficients : ndarray
            Contains geometry and material properties. It is a 0-rank tensor (or scalar) field.
        nurbs_weights : ndarray
            The weights of the B-spline basis to compute the NURBS basis.

        """
        raise NotImplementedError("To implement in children")

    @staticmethod
    @abstractmethod
    def assemble_scalar_gradu_force(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> np.ndarray:
        """
        Computes the force-like array F using matrix-free algorithms.

        Here the terms of F results from
        'int grad(N_A(x)) . c(x) dx' <- in the hypercube [0, 1]^d
        where N_A are basis functions in the same parametric space.

        Parameters
        ----------
        quadrule_list : List[QuadratureRule]
            List of quadrature rules to build the basis functions and to compute the integral.
        coefficients : ndarray
            Contains geometry and material properties. It is a 0-rank tensor (or scalar) field.
        nurbs_weights : ndarray
            The weights of the B-spline basis to compute the NURBS basis.

        """
        raise NotImplementedError("To implement in children")

    @staticmethod
    @abstractmethod
    def compute_mf_scalar_u_v(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        array_in: np.ndarray,
        time_ders: Optional[Sequence[int]] = None,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> np.ndarray:
        """
        Computes the matrix-vector product M @ v using matrix-free algorithms.

        Here the terms of M (also called mass matrix) results from
        'int N_A(x) c(x) N_B(x) dx' <- in the hypercube [0, 1]^d
        where N_A and N_B are basis functions in the same parametric space.

        To generalize the method, basis funcitons may also include time deirvatives
        N_A(x, t) = N^p_t(t) x N_1(x_1) x N_2(x_2) x ... x N_d(x_d)
        if p = 0, there is no derivative, p = 1, it has been derived once, and so on.

        Parameters
        ----------
        quadrule_list : List[QuadratureRule]
            List of quadrature rules to build the basis functions and to compute the integral.
        coefficients : ndarray
            Contains geometry and material properties. It is a 0-rank tensor (or scalar) field.
        array_in : ndarray
            The vector to be multiplied.
        time_ders : Sequence[int]
            Tuple of size 2, the first element sets the derivative of N_A,
            the second elements sets the derivative of N_B.
        nurbs_weights : ndarray
            The weights of the B-spline basis to compute the NURBS basis.
        """
        raise NotImplementedError("To implement in children")

    @staticmethod
    @abstractmethod
    def compute_mf_scalar_gradu_gradv(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        array_in: np.ndarray,
        time_ders: Optional[Sequence[int]] = None,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> np.ndarray:
        """
        Computes the matrix-vector product M @ v using matrix-free algorithms.

        Here the terms of M (also called stiffness matrix) results from
        'int grad(N_A(x)) [c(x) . grad(N_B(x))] dx' <- in the hypercube [0, 1]^d.
        where N_A and N_B are basis functions in the same parametric space.

        To generalize the method, basis funcitons may also include time deirvatives
        N_A(x, t) = N^p_t(t) x N_1(x_1) x N_2(x_2) x ... x N_d(x_d).
        If p = 0, there is no derivative, p = 1, it has been derived once, and so on.

        Parameters
        ----------
        quadrule_list : List[QuadratureRule]
            List of quadrature rules to build the basis functions and to compute the integral.
        coefficients : ndarray
            Contains geometry and material properties. It is a 2-rank tensor field.
        array_in : ndarray
            The vector to be multiplied.
        time_ders : Sequence[int]
            Tuple of size 2, the first element sets the derivative of N_A,
            the second elements sets the derivative of N_B.
        nurbs_weights : ndarray
            The weights of the B-spline basis to compute the NURBS basis.
        """
        raise NotImplementedError("To implement in children")

    @staticmethod
    @abstractmethod
    def compute_mf_scalar_gradu_v(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        array_in: np.ndarray,
        time_ders: Optional[Sequence[int]] = None,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> np.ndarray:
        """
        Computes the matrix-vector product M @ v using matrix-free algorithms.

        Here the terms of M (also called advection matrix) from
        'int [grad(N_A(x)) . c(x)] . N_B(x) dx' <- in the hypercube [0, 1]^d
        where N_A and N_B are basis functions in the same parametric space.

        To generalize the method, basis funcitons may also include time deirvatives
        N_A(x, t) = N^p_t(t) x N_1(x_1) x N_2(x_2) x ... x N_d(x_d)
        if p = 0, there is no derivative, p = 1, it has been derived once, and so on.

        Parameters
        ----------
        quadrule_list : List[QuadratureRule]
            List of quadrature rules to build the basis functions and to compute the integral.
        coefficients : ndarray
            Contains geometry and material properties. It is a 1-rank tensor field.
        array_in : ndarray
            The vector to be multiplied.
        time_ders : Sequence[int]
            Tuple of size 2, the first element sets the derivative of N_A,
            the second elements sets the derivative of N_B.
        nurbs_weights : ndarray
            The weights of the B-spline basis to compute the NURBS basis.
        """
        raise NotImplementedError("To implement in children")

    @staticmethod
    @abstractmethod
    def compute_mf_scalar_u_gradv(
        quadrule_list: Sequence[QuadratureRule],
        coefficients: np.ndarray,
        array_in: np.ndarray,
        time_ders: Optional[Sequence[int]] = None,
        nurbs_weights: np.ndarray = np.array([]),
    ) -> np.ndarray:
        """
        Computes the matrix-vector product M @ v using matrix-free algorithms.

        Here the terms of M (also called advection matrix transposed) results from
        'int N_A(x) [c(x) . grad(N_B(x))] dx' <- in the hypercube [0, 1]^d
        where N_A and N_B are basis functions in the same parametric space.

        To generalize the method, basis funcitons may also include time deirvatives
        N_A(x, t) = N^p_t(t) x N_1(x_1) x N_2(x_2) x ... x N_d(x_d)
        if p = 0, there is no derivative, p = 1, it has been derived once, and so on.

        Parameters
        ----------
        quadrule_list : List[QuadratureRule]
            List of quadrature rules to build the basis functions and to compute the integral.
        coefficients : ndarray
            Contains geometry and material properties. It is a 1-rank tensor field.
        array_in : ndarray
            The vector to be multiplied.
        time_ders : Sequence[int]
            Tuple of size 2, the first element sets the derivative of N_A,
            the second elements sets the derivative of N_B.
        nurbs_weights : ndarray
            The weights of the B-spline basis to compute the NURBS basis.
        """
        raise NotImplementedError("To implement in children")
