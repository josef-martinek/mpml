from core.linsolver.linsorver_base import LinsorverBase
from petsc4py import PETSc


class minres(LinsorverBase):

    atol = 1e-50

    @staticmethod
    def solve_system(A, b, rtol):
        x = A.createVecRight()

        ksp = PETSc.KSP().create()
        ksp.setOperators(A)  # Set the matrix A as the operator
        ksp.setType("minres")  # Set the solver type to MINRES
        ksp.getPC().setType("none")  # No preconditioner
        ksp.setTolerances(rtol=rtol, atol=minres.atol)

        PETSc.Log.begin()
        initial_flops = PETSc.Log.getFlops()
        ksp.solve(b, x)
        final_flops = PETSc.Log.getFlops()
        flops_performed = final_flops - initial_flops
        # Get solver information

        num_it = ksp.getIterationNumber()  # Number of iterations
        conv_reason = ksp.getConvergedReason()  # Convergence reason
        if conv_reason != PETSc.KSP.ConvergedReason.CONVERGED_RTOL:
            raise ValueError(f"Solver did not converge by achieving specified relative residual tolerance. Reason: {conv_reason}")

        return x, num_it, conv_reason, flops_performed