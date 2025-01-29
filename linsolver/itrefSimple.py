import numpy as np
from linsolver.itref import itref
from pychop.chop import chop

class PrecisionSetting:
    setting = {"q": 'q43', "h": np.half, "s": np.single, "d": np.double}


class itrefSimple(itref):
    
    def __init__(self, eps_f, eps_s, eps, eps_r):
        if "q" in [eps_s, eps, eps_r]:
            raise ValueError("Quarter precision only supported for factorisation.")
        self._eps_r = PrecisionSetting.setting[eps_r]
        self._eps = PrecisionSetting.setting[eps]
        self._eps_s = PrecisionSetting.setting[eps_s]
        self._eps_f = PrecisionSetting.setting[eps_f]
    
    @staticmethod
    def forward_substitution(L, b, prec):
        """
        Perform forward substitution to solve Lx = b, where L is a lower triangular matrix.

        Parameters:
            L (np.ndarray): Lower triangular matrix (n x n).
            b (np.ndarray): Right-hand side vector (n,).

        Returns:
            x (np.ndarray): Solution vector (n,).
        """
        if prec != 'q43':
            return super(itrefSimple, itrefSimple).forward_substitution(L, b, prec)
        else:
            cp = chop(prec=prec)
            n = L.shape[0]
            x = np.zeros_like(b, dtype=L.dtype)  # Solution vector with same dtype as L

            x[0] = cp(b[0]/L[0,0])

            for i in range(1, n):
                if L[i, i] == 0:
                    raise ValueError("Matrix is singular!")
                dotproduct = 0
                for j in range(i):  # Loop over the elements of the slice L[i, :i] and x[:i]
                    dotproduct = cp(dotproduct + cp(L[i][j] * x[j]))
                x[i] = cp(cp(b[i] - dotproduct) / L[i, i])

        return x


    @staticmethod
    def backward_substitution(U, b, prec):
        """
        Perform backward substitution to solve Ux = b, where U is an upper triangular matrix.

        Parameters:
            U (np.ndarray): Upper triangular matrix (n x n).
            b (np.ndarray): Right-hand side vector (n,).

        Returns:
            x (np.ndarray): Solution vector (n,).
        """
        if prec != 'q43':
            return super(itrefSimple, itrefSimple).backward_substitution(U, b, prec)
        else:
            cp = chop(prec=prec)
            n = U.shape[0]
            x = np.zeros_like(b, dtype=U.dtype)  # Solution vector with same dtype as U

            x[n - 1] = cp(b[n - 1]/U[n - 1,n - 1])

            for i in range(n - 2, -1, -1):
                if U[i, i] == 0:
                    raise ValueError("Matrix is singular!")
                dotproduct = 0
                for j in range(i+1, n):
                    dotproduct = cp(dotproduct + cp(U[i][j] * x[j]))
                x[i] = (b[i] - dotproduct) / U[i, i]

            return x
    
    def _get_cholesky_factor(self, A, prec):
        if prec != 'q43':
            if prec == np.half:
                return super().cholesky_factorisation(A, prec)
            else:
                return np.linalg.cholesky(A)
        else:
            return self.cholesky_factorisation(A, prec)
        
    @staticmethod    
    def cholesky_factorisation(A, prec):
        """
        Perform Cholesky factorization on a positive-definite symmetric matrix A.
        
        Parameters:
        A (ndarray): A symmetric, positive-definite matrix.
        
        Returns:
        L (ndarray): A lower triangular matrix such that A = L @ L.T.
        """
        # Ensure the matrix is square
        cp = chop(prec=prec)
        original_type = A.dtype
        A = A.astype(np.double)
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
                    s = 0
                    for k in range(j):
                        s = cp(s + cp(L[i, k] ** 2))
                    L[i, j] = cp(np.sqrt(cp(A[i, i] - s)))
                else:  # Off-diagonal elements
                    s = 0
                    for k in range(j):
                        s = cp(s + cp(L[i, k]*L[j, k]))
                    L[i, j] = cp(cp(A[i, j] - s) / L[j, j])
        
        return L.astype(original_type)