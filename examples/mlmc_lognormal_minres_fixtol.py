from model.model_base import ModelEvaluationBase
from examples.mlmc_lognormal import LognormalPDEModel, LognormalPDEEvaluation
from dolfinx.fem import form, apply_lifting, Function, functionspace
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import logging
from examples.lognormal_pde_setup import PDEwLognormalRandomCoeff as PDE
from linsolver.minres import minres


class MLLognormalPDEModelMinres(LognormalPDEModel):

    def __init__(self, visualise=False):
        super().__init__(visualise)
        self._comp_tol = 1e-10

    def evaluate(self, level, sample) -> ModelEvaluationBase:
        hl = self.get_hl(level)
        a, L, bc = PDE.setup_fenics_problem(hl, sample, self._decay_rate_q)
        a = form(a)
        L = form(L)
        A = assemble_matrix(a, bcs=[bc])
        A.assemble()
        b = assemble_vector(L)
        b.assemble()
        apply_lifting(b.array, [a], bcs=[[bc]])
        bc.set(b.array)

        x, num_it, conv_reason, flops_performed = minres.solve_system(A=A, b=b, rtol=self._comp_tol)

        logging.trace(f'Iteration number on level {level} is {num_it}')
        logging.trace(f'Convergence reason: {conv_reason}')
        logging.trace(f'Number of flops per iteration: {flops_performed/num_it}\n')
       
        msh = PDE.get_mesh(hl)
        uh = Function(functionspace(msh, ("Lagrange", 1)))
        uh.x.array[:] = x.array[:]
        qoi = self._get_qoi_from_solution(uh, level)
        evaluation_cost = self._get_eval_cost(flops_performed)
        return LognormalPDEEvaluation(qoi, evaluation_cost)
    
    def _get_eval_cost(self, nflops_per_solve):
        return nflops_per_solve