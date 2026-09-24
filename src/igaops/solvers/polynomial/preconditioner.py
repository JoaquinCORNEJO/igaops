from typing import Any, Optional, List, Tuple

import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator


class GMRESPolynomialPreconditioner(LinearOperator):
    """Polynomial preconditioner p(A) ~= A^{-1} constructed from GMRES.

    The GMRES residual polynomial is

        r_m(z) = prod_i (1 - z / theta_i),

    where theta_i are the harmonic Ritz values associated with an m-step
    Arnoldi process.

    The polynomial preconditioner is

        p_{m-1}(z) = (1 - r_m(z)) / z.

    It is applied without explicitly forming the polynomial coefficients,
    using the root representation

        r <- (I - A / theta) r
        y <- y + r_old / theta.

    For a real matrix, complex-conjugate roots are applied together so that
    the preconditioner remains entirely real.

    Parameters
    ----------
    A : array_like or LinearOperator
        Matrix or linear operator to precondition.
    degree : int, default=10
        Degree of the GMRES residual polynomial.
    b0 : array_like, optional
        Starting vector used to construct the Arnoldi process. If None,
        a reproducible random vector is generated.
    breakdown_tol : float, default=1e-14
        Relative tolerance used to detect Arnoldi breakdown.
    root_tol : float, default=1e-12
        Relative tolerance for detecting harmonic Ritz values too close
        to zero.

    Notes
    -----
    The polynomial is constructed once during initialization and is then
    fixed. Applying the preconditioner requires one A-matvec per real root
    and two A-matvecs per complex-conjugate pair.
    """

    def __init__(
        self,
        A: Any,
        degree: int = 10,
        b0: Optional[np.ndarray] = None,
        breakdown_tol: float = 1e-14,
        root_tol: float = 1e-12,
    ):
        A = aslinearoperator(A)

        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be square.")

        if not isinstance(degree, (int, np.integer)) or degree < 1:
            raise ValueError("degree must be a positive integer.")

        if not np.issubdtype(A.dtype, np.number):
            raise TypeError("A must have a numerical dtype.")

        # Ensure that all Arnoldi operations are performed in floating point.
        if np.issubdtype(A.dtype, np.complexfloating):
            dtype = np.result_type(A.dtype, np.complex128)
        else:
            dtype = np.result_type(A.dtype, np.float64)

        self.A = A
        self.degree_requested = int(degree)
        self.breakdown_tol = breakdown_tol
        self.root_tol = root_tol

        self._is_real = not np.issubdtype(dtype, np.complexfloating)

        super().__init__(
            dtype=dtype,
            shape=A.shape,
        )

        n = A.shape[0]
        self.degree = min(self.degree_requested, n)

        # ---------------------------------------------------------------
        # Starting vector
        # ---------------------------------------------------------------
        if b0 is None:
            rng = np.random.default_rng(0)

            if self._is_real:
                b0 = rng.standard_normal(n)
            else:
                b0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)

        b0 = np.asarray(b0)

        if b0.ndim != 1 or b0.size != n:
            raise ValueError(f"b0 must be a vector of length {n}.")

        b0 = b0.astype(dtype, copy=False)

        beta = np.linalg.norm(b0)

        if beta == 0:
            raise ValueError("b0 must be nonzero.")

        # ---------------------------------------------------------------
        # Construct harmonic Ritz values
        # ---------------------------------------------------------------
        self.roots = self._harmonic_ritz(b0)

        if self.roots.size == 0:
            raise RuntimeError("The Arnoldi process produced no harmonic Ritz values.")

        # ---------------------------------------------------------------
        # Check roots
        # ---------------------------------------------------------------
        root_scale = max(1.0, np.max(np.abs(self.roots)))
        zero_tol = self.root_tol * root_scale

        if np.any(np.abs(self.roots) <= zero_tol):
            bad_roots = self.roots[np.abs(self.roots) <= zero_tol]

            raise np.linalg.LinAlgError(
                "A harmonic Ritz value is too close to zero. "
                "The polynomial would contain a factor 1/theta and "
                "would be numerically unstable. "
                f"Problematic roots: {bad_roots}"
            )

        # ---------------------------------------------------------------
        # Leja ordering
        # ---------------------------------------------------------------
        self._steps = self._leja_steps(self.roots)

        # Number of roots represented by the steps.
        self.degree_actual = len(self.roots)

    # ===================================================================
    # Arnoldi / harmonic Ritz construction
    # ===================================================================

    def _harmonic_ritz(self, b0: np.ndarray) -> np.ndarray:
        """Compute harmonic Ritz values from an Arnoldi process."""

        n = self.shape[0]
        m = self.degree

        V = np.zeros(
            (n, m + 1),
            dtype=self.dtype,
        )

        H = np.zeros(
            (m + 1, m),
            dtype=self.dtype,
        )

        V[:, 0] = b0 / np.linalg.norm(b0)

        actual_m = m

        for j in range(m):
            w = self.A.matvec(V[:, j])
            w = np.asarray(w, dtype=self.dtype)

            # Modified Gram-Schmidt + one reorthogonalization.
            for _ in range(2):
                h = V[:, : j + 1].conj().T @ w
                w -= V[:, : j + 1] @ h
                H[: j + 1, j] += h

            beta = np.linalg.norm(w)
            H[j + 1, j] = beta

            # Relative Arnoldi breakdown test.
            column_scale = max(
                1.0,
                np.linalg.norm(H[: j + 1, j]),
            )

            if beta <= self.breakdown_tol * column_scale:
                actual_m = j + 1
                break

            V[:, j + 1] = w / beta

        else:
            actual_m = m

        # ---------------------------------------------------------------
        # Harmonic Ritz values
        #
        # Hbar = H_m + |h_{m+1,m}|^2
        #              (H_m^*)^{-1} e_m e_m^*
        # ---------------------------------------------------------------

        Hm = H[:actual_m, :actual_m]

        if actual_m == 1:
            hlast = H[1, 0]
        else:
            hlast = H[actual_m, actual_m - 1]

        e = np.zeros(
            actual_m,
            dtype=self.dtype,
        )
        e[-1] = 1.0

        try:
            f = np.linalg.solve(
                Hm.conj().T,
                e,
            )
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "The projected Arnoldi matrix H_m is singular or "
                "numerically singular, so harmonic Ritz values could "
                "not be computed."
            ) from exc

        harmonic_matrix = Hm + abs(hlast) ** 2 * np.outer(f, e.conj())

        roots = np.linalg.eigvals(harmonic_matrix)

        # For a real A, numerical eigensolvers may return tiny imaginary
        # parts for roots that should be exactly real.
        if self._is_real:
            scale = max(1.0, np.max(np.abs(roots)))
            imag_tol = 100 * np.finfo(float).eps * scale

            roots = np.array(
                [z.real if abs(z.imag) <= imag_tol else z for z in roots],
                dtype=self.dtype,
            )

        return roots

    # ===================================================================
    # Root ordering
    # ===================================================================
    def _leja_steps(self, roots: np.ndarray) -> List[Tuple[str, float]]:
        """Leja-order real roots and complex-conjugate root pairs."""

        if not self._is_real:
            candidates = [("r", theta) for theta in roots]
        else:
            scale = max(
                1.0,
                max(abs(theta) for theta in roots),
            )

            imag_tol = 100 * np.finfo(float).eps * scale

            candidates = []

            for theta in roots:
                if abs(theta.imag) <= imag_tol:
                    candidates.append(("r", float(theta.real)))
                elif theta.imag > imag_tol:
                    candidates.append(("c", complex(theta)))

        chosen = []
        steps = []

        while candidates:
            if not chosen:
                # Start with the root/pair having largest magnitude.
                j = int(np.argmax([abs(theta) for _, theta in candidates]))
            else:
                # Actual roots already selected, including conjugates.
                previous = []

                for kind, theta in chosen:
                    previous.append(theta)

                    if kind == "c":
                        previous.append(np.conj(theta))

                scores = []

                for kind, theta in candidates:
                    current = [theta]

                    if kind == "c":
                        current.append(np.conj(theta))

                    score = 0.0

                    for z in current:
                        for q in previous:
                            distance = abs(z - q)

                            score += np.log(
                                max(
                                    distance,
                                    np.finfo(float).tiny,
                                )
                            )

                    scores.append(score)

                j = int(np.argmax(scores))

            item = candidates.pop(j)
            chosen.append(item)
            steps.append(item)

        return steps

    # ===================================================================
    # Polynomial application
    # ===================================================================

    def _matvec(self, x):
        """Apply y = p(A)v using the root representation."""

        x = np.asarray(x).reshape(-1)

        if x.size != self.shape[1]:
            raise ValueError(
                f"Expected vector of size {self.shape[1]}, " f"got {x.size}."
            )

        x = x.astype(
            self.dtype,
            copy=False,
        )

        y = np.zeros_like(
            x,
            dtype=self.dtype,
        )

        # Current residual r.
        r = x.copy()

        for kind, theta in self._steps:

            if kind == "r":
                # -------------------------------------------------------
                # Real root:
                #
                # y_new = y + r/theta
                # r_new = r - A r/theta
                # -------------------------------------------------------
                Ar = self.A.matvec(r)
                Ar = np.asarray(
                    Ar,
                    dtype=self.dtype,
                )

                y += r / theta
                r -= Ar / theta

            else:
                # -------------------------------------------------------
                # Complex conjugate pair:
                #
                # theta = a + ib
                # c = |theta|^2
                #
                # (I - A/theta)(I - A/conj(theta))
                #
                # = I - 2a/c A + 1/c A^2
                #
                # and the corresponding contribution to p(A)v is
                #
                # 2a/c r - 1/c Ar.
                # -------------------------------------------------------
                a = theta.real
                c = abs(theta) ** 2

                Ar = self.A.matvec(r)
                Ar = np.asarray(
                    Ar,
                    dtype=self.dtype,
                )

                A2r = self.A.matvec(Ar)
                A2r = np.asarray(
                    A2r,
                    dtype=self.dtype,
                )

                y += (2.0 * a / c) * r - Ar / c

                r = r - (2.0 * a / c) * Ar + A2r / c

        return y
