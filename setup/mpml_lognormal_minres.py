from core.model.model_base import MPMLModel, ModelEvaluationBase
from setup.mlmc_lognormal import LognormalPDEModel, LognormalPDEEvaluation
from dolfinx.fem import form, apply_lifting, Function, functionspace
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import logging
from setup.lognormal_pde_setup import PDEwLognormalRandomCoeff as PDE
from core.linsolver.minres import minres, minresMonitor


class MPLognormalPDEModelMinres(MPMLModel, LognormalPDEModel):

    def evaluate(self, level, sample, comp_tol) -> ModelEvaluationBase:
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

        x, num_it, conv_reason, flops_performed = minres.solve_system(A=A, b=b, rtol=comp_tol)

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


class LognormalMinresMonitor(LognormalPDEModel):

    def evaluate(self):
        raise NotImplementedError("This class cannot be used in MLMC, only for computing solution stats.")

    def evaluate_monitor(self, level, sample):
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

        # Solve the system and monitor intermediate solutions
        x, num_it, conv_reason, iterates, rel_residuals = minresMonitor.solve_system(A=A, b=b)

        logging.trace(f'Iteration number on level {level} is {num_it}')
        logging.trace(f'Convergence reason: {conv_reason}')
        logging.trace(f'Relative residuals: {rel_residuals}')

        # Compute QoI for all intermediate iterates
        qois = []
        msh = PDE.get_mesh(hl)
        for xi in iterates:
            uh_i = Function(functionspace(msh, ("Lagrange", 1)))
            uh_i.x.array[:] = xi.array[:]
            qoi_i = self._get_qoi_from_solution(uh_i, level)
            qois.append(qoi_i)

        return {
            "qois_per_iter": qois,
            "rel_residuals": rel_residuals,
        }
