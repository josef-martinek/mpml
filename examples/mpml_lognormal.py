import dolfinx.fem.petsc
from model.model_base import MPMLModel, ModelEvaluationBase
from examples.mlmc_lognormal import LognormalPDEModel, LognormalPDEEvaluation
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
import logging


class MPLognormalPDEModel(MPMLModel, LognormalPDEModel):

    def evaluate(self, level, sample, comp_tol) -> ModelEvaluationBase:
        a, L, bc = self._setup_fenicsx_problem(level, sample)
        #problem = LinearProblem(a, L, bcs=[bc], petsc_options=self._get_petsc_options(comp_tol))
        #PETSc.Log.begin()
        uh = problem.solve()
        #total_flops = PETSc.Log.getFlops()
        ksp = problem.solver
        num_it = ksp.getIterationNumber()
        conv_reason = ksp.getConvergedReason()
        logging.debug(f'Iteration number on level {level} is {num_it}')
        logging.debug(f'Convergence reason: {conv_reason}')
        #logging.debug(f'Number of flops per iteration: {total_flops/num_it}')
        if conv_reason != 2: raise ValueError("Solver converged according to a different criterion than relative residual")
        qoi = self._get_qoi_from_solution(uh, level)
        if self._visualise:
            self._plot_solution(uh, level)
        evaluation_cost = self._get_eval_cost(level)
        return LognormalPDEEvaluation(qoi, evaluation_cost)
    
    def _get_petsc_options(self, comp_tol):
        return {"ksp_type": "minres", "pc_type": "none", "ksp_rtol": comp_tol}
    
    #TODO Adjust cost for MINRES!!!