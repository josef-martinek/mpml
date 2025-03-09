from core.model.model_base import MPMLModel, ModelEvaluationBase
from setup.mlmc_lognormal import LognormalPDEModel, LognormalPDEEvaluation
from dolfinx.fem import form, apply_lifting, Function, functionspace
from dolfinx.fem import assemble_matrix, assemble_vector, locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary
import logging
from setup.lognormal_pde_setup import PDEwLognormalRandomCoeff as PDE
import numpy as np
from core.linsolver.itref_qp import itrefQP
from core.sample.sample_base import SampleBase
from setup.mpml_lognormal_itref import MPLognormalPDEModelItref
import time


class LognormalPDESampleQP(SampleBase):

    def __init__(self, rng: np.random.Generator):
        self._s = 1
        self._std = 1
        self._rng = rng

    def draw(self):
        return self._std*self._rng.standard_normal(self._s)


class MPLognormalPDEModelItrefQP(MPLognormalPDEModelItref):

    def evaluate(self, level, sample, comp_tol) -> ModelEvaluationBase:
        hl = self.get_hl(level)
        a, L, bc = PDE.setup_fenics_problem(hl, sample, self._decay_rate_q)
        a = form(a)
        L = form(L)
        A = assemble_matrix(a, bcs=[bc])
        b = assemble_vector(L)
        apply_lifting(b.array, [a], bcs=[[bc]])
        bc.set(b.array)

        # Find the Dirichlet DoFs
        dirichlet_dofs = bc.dof_indices()[0]
        A_full = self._csr_mat_to_ndarray(A)
        b_full = self._vec_to_ndarray(b)

        # Determine all active degrees of freedom (not in `dirichlet_dofs`)
        total_dofs = A_full.shape[0]
        active_dofs = np.setdiff1d(np.arange(total_dofs), dirichlet_dofs)

        # Extract the submatrix and subvector
        A_reduced = A_full[np.ix_(active_dofs, active_dofs)]  # Rows and columns for active DoFs
        b_reduced = b_full[active_dofs]  # Only rows for active DoFs

        itref_l = self._get_itref(hl)
        start_time = time.time()
        x_reduced, num_it, rel_res = itref_l.solve_system(A=A_reduced, b=b_reduced, rtol=comp_tol)
        end_time = time.time()

        logging.trace(f'Iterative refinement runtime: {end_time - start_time}')
        logging.trace(f'Itref iteration number on level {level} with hl {hl} is {num_it}')
        logging.trace(f'Computational tolerance: {comp_tol}')
        logging.trace(f'Relative residual norms: {rel_res}\n')
        if rel_res[-1] > comp_tol:
            logging.warning(f'Itref not converged within given max number of iterations. Rel. residual: {rel_res[-1]}. Tolerance: {comp_tol}.\n')

        # Expand the reduced solution back to the full system (if needed)
        x_full = np.zeros(total_dofs)
        x_full[active_dofs] = x_reduced[:]
       
        msh = PDE.get_mesh(hl)
        uh = Function(functionspace(msh, ("Lagrange", 1)))
        uh.x.array[:] = x_full[:]
        qoi = self._get_qoi_from_solution(uh, level)
        evaluation_cost = self._get_eval_cost(level=level)
        return LognormalPDEEvaluation(qoi, evaluation_cost)
    
    @staticmethod
    def _get_itref(hl):
        if np.allclose(hl, 1/4):
            return itrefQP("q", "h", "h", "h")
        elif np.allclose(hl, 1/8):
            return itrefQP("h", "h", "s", "s")
        elif np.allclose(hl, 1/16):
            return itrefQP("h", "h", "s", "s")
        else:
            return itrefQP("h", "s", "s", "s")