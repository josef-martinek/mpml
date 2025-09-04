import numpy as np
from core.linsolver.linsorver_base import LinsorverBase

class itref(LinsorverBase):
    
    def __init__(self, eps_f: np.floating, eps_s: np.floating, eps: np.floating, eps_r: np.floating):
        self._eps_r = eps_r
        self._eps = eps
        self._eps_s = eps_s
        self._eps_f = eps_f

    def solve_system(self, A: np.ndarray, b: np.ndarray, rtol, max_iter=10):
        """
        Perform iterative refinement to solve Ax = b using three precisions (half, single, and double).

        Parameters:
        - A (np.ndarray): Coefficient matrix (assumed SPD).
        - TODO

        Returns:
        - x (np.ndarray): Refined solution
        - TODO
        """
        A = A.astype(self._eps)
        b = b.astype(self._eps)

        # Perform Cholesky decomposition in single precision
        L_f = self._get_cholesky_factor(A, prec=self._eps_f)
        y = self.forward_substitution(L=L_f, b=b, prec=self._eps_f)
        x = self.backward_substitution(U=np.transpose(L_f), b=y, prec=self._eps_f)
        x = x.astype(self._eps)

        rel_res = []
        for i in range(max_iter):
            # Step 1: Compute residual in double precision
            r = b.astype(self._eps_r) - A.astype(self._eps_r) @ x.astype(self._eps_r)
            rel_res.append(np.linalg.norm(r)/np.linalg.norm(b))

            if (np.linalg.norm(r)/np.linalg.norm(b)) < rtol:
                return x.astype(np.double), i, np.array(rel_res).astype(np.double)

            # Step 2: Convert residual to half precision
            r = r.astype(self._eps_s)

            # Step 3: Solve for correction in single precision
            y = self.forward_substitution(L=L_f, b=r, prec=self._eps_s)
            x_update = self.backward_substitution(np.transpose(L_f), b=y, prec=self._eps_s)

            x_update = x_update.astype(self._eps)
            x = x.astype(self._eps) + x_update.astype(self._eps)

        return x.astype(np.double), i, np.array(rel_res).astype(np.double)
    
    @staticmethod
    def forward_substitution(L, b, prec: np.floating):
        """
        Perform forward substitution to solve Lx = b, where L is a lower triangular matrix.

        Parameters:
            L (np.ndarray): Lower triangular matrix (n x n).
            b (np.ndarray): Right-hand side vector (n,).

        Returns:
            x (np.ndarray): Solution vector (n,).
        """
        L = L.astype(prec)
        b = b.astype(prec)
        n = L.shape[0]
        x = np.zeros_like(b, dtype=L.dtype)  # Solution vector with same dtype as L

        x[0] = b[0]/L[0,0]

        for i in range(1, n):
            if L[i, i] == 0:
                raise ValueError("Matrix is singular!")
            x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]

        return x


    @staticmethod
    def backward_substitution(U, b, prec: np.floating):
        """
        Perform backward substitution to solve Ux = b, where U is an upper triangular matrix.

        Parameters:
            U (np.ndarray): Upper triangular matrix (n x n).
            b (np.ndarray): Right-hand side vector (n,).

        Returns:
            x (np.ndarray): Solution vector (n,).
        """
        U = U.astype(prec)
        b = b.astype(prec)
        n = U.shape[0]
        x = np.zeros_like(b, dtype=U.dtype)  # Solution vector with same dtype as U

        x[n - 1] = b[n - 1]/U[n - 1,n - 1]

        for i in range(n - 2, -1, -1):
            if U[i, i] == 0:
                raise ValueError("Matrix is singular!")
            x[i] = (b[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

        return x
    
    def _get_cholesky_factor(self, A, prec: np.floating):
        A = A.astype(prec)
        if self._eps_f == np.half:
            return self.cholesky_factorisation(A, prec)
        else:
            return np.linalg.cholesky(A)
        
    @staticmethod    
    def cholesky_factorisation(A, prec: np.floating):
        """
        Perform Cholesky factorization on a positive-definite symmetric matrix A.
        
        Parameters:
        A (ndarray): A symmetric, positive-definite matrix.
        
        Returns:
        L (ndarray): A lower triangular matrix such that A = L @ L.T.
        """
        A = A.astype(prec)
        # Ensure the matrix is square
        n, m = A.shape
        if n != m:
            raise ValueError("Matrix A must be square")
        
        # Ensure the matrix is symmetric
        if not np.allclose(A, A.T):
            raise ValueError("Matrix A must be symmetric")
        
        # Initialize L as a zero matrix
        L = np.zeros_like(A, dtype=A.dtype)
        
        # Perform Cholesky factorization
        for i in range(n):
            for j in range(i + 1):
                if i == j:  # Diagonal elements
                    L[i, j] = np.sqrt(A[i, i] - np.sum(L[i, :j] ** 2))
                else:  # Off-diagonal elements
                    L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
        
        return L