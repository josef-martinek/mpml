import dolfinx.fem.petsc
from model.model_base import MPMLModel, ModelEvaluationBase
from examples.mlmc_lognormal import LognormalPDEModel, LognormalPDEEvaluation
from dolfinx.fem import form, apply_lifting, Function, functionspace
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector
from petsc4py import PETSc
import logging


class MPLognormalPDEModel(MPMLModel, LognormalPDEModel):

    def evaluate(self, level, sample, comp_tol) -> ModelEvaluationBase:
        a, L, bc, msh = self._setup_fenicsx_problem(level, sample)
        #problem = LinearProblem(a, L, bcs=[bc], petsc_options=self._get_petsc_options(comp_tol))
        #uh = problem.solve()
        a = form(a)
        L = form(L)
        A = assemble_matrix(a, bcs=[bc])
        A.assemble()
        b = assemble_vector(L)
        b.assemble()
        apply_lifting(b.array, [a], bcs=[[bc]])
        bc.set(b.array)
        x = A.createVecRight()

        # Create the Krylov Subspace Solver (KSP)
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)  # Set the matrix A as the operator
        ksp.setType("minres")  # Set the solver type to MINRES
        ksp.getPC().setType("none")  # No preconditioner

        # Set the convergence tolerance
        #comp_tol = 1e-6  # Example relative tolerance
        #ksp.setTolerances(rtol=comp_tol)
        #ksp.setTolerances(rtol=1e-8)
        #ksp.setTolerances(rtol=1e-16)

        # Solve the system
        # Get the initial FLOP count
        PETSc.Log.begin()
        initial_flops = PETSc.Log.getFlops()

        # Solve the linear system
        ksp.solve(b, x)

        # Get the total FLOPs performed since the initial reading
        final_flops = PETSc.Log.getFlops()
        flops_performed = final_flops - initial_flops
        # Get solver information
        num_it = ksp.getIterationNumber()  # Number of iterations
        conv_reason = ksp.getConvergedReason()  # Convergence reason
        if conv_reason != PETSc.KSP.ConvergedReason.CONVERGED_RTOL:
            raise ValueError(f"Solver did not converge by achieving specified relative residual tolerance. Reason: {conv_reason}")

        logging.debug(f'Iteration number on level {level} is {num_it}')
        logging.debug(f'Convergence reason: {conv_reason}')
        logging.debug(f'Number of flops per iteration: {flops_performed/num_it}\n')
        if conv_reason != 2: raise ValueError("Solver converged according to a different criterion than relative residual")
        uh = Function(functionspace(msh, ("Lagrange", 1)))
        uh.x.array[:] = x.array[:]
        qoi = self._get_qoi_from_solution(uh, level)
        if self._visualise:
            self._plot_solution(uh, level)
        evaluation_cost = self._get_eval_cost(flops_performed)
        return LognormalPDEEvaluation(qoi, evaluation_cost)
    
    def _get_petsc_options(self, comp_tol):
        return {"ksp_type": "minres", "pc_type": "none", "ksp_rtol": comp_tol}
    
    def _get_eval_cost(self, nflops_per_solve):
        return nflops_per_solve