import george
import numpy as np
import scipy.optimize as op
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.real_simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.real_polynomial_mutation import PolynomialMutation
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.operators.survival.fitness_survival import FitnessSurvival

from pymop.problem import Problem

from pysurrogate.surrogate import Surrogate


class GPGeorge(Surrogate):
    def __init__(self, kernel, opt):
        Surrogate.__init__(self)
        self.kernel = kernel
        self.opt = opt
        self.model = None
        self.F = None

    def _predict(self, X):
        return self.model.predict(self.F, X, return_var=True)

    def _fit(self, X, F, data):

        self.F = F
        n_var = X.shape[1]

        if self.kernel == "linear":
            kernel = george.kernels.LinearKernel(order=2, log_gamma2=0.2, ndim=n_var)
        elif self.kernel == "expsqrt":
            kernel = george.kernels.ExpSquaredKernel(metric=np.ones(n_var), ndim=n_var)
        elif self.kernel == "rational_quad":
            kernel = george.kernels.RationalQuadraticKernel(log_alpha=0.2, metric=np.ones(n_var), ndim=n_var)
        elif self.kernel == "exp":
            kernel = george.kernels.ExpKernel(metric=np.ones(n_var), ndim=n_var)
        elif self.kernel == "polynomial":
            kernel = george.kernels.PolynomialKernel(metric=np.ones(n_var))
        else:
            raise ValueError("Parameter %s for kernel unknown." % self.kernel)

        gp = george.GP(kernel, fit_mean=True)
        gp.compute(X)

        def nll(p):
            gp.set_parameter_vector(p)
            ll = gp.log_likelihood(F, quiet=True)
            return -ll if np.isfinite(ll) else 1e25

        def grad_nll(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(F, quiet=True)

        if 'expensive' in data and data['expensive']:
            n_restarts = 20
        else:
            n_restarts = 5

        n_hyper_var = len(gp.get_parameter_vector())

        # print(gp.get_parameter_bounds(include_frozen=False))

        class HyperparameterProblem(Problem):

            def __init__(self, **kwargs):
                Problem.__init__(self, **kwargs)
                self.n_var = n_hyper_var
                self.n_constr = 0
                self.n_obj = 1
                self.func = self.evaluate_
                self.xl = 0 * np.ones(self.n_var)
                self.xu = 10 * np.ones(self.n_var)

            def evaluate_(self, x, f):
                for i in range(x.shape[0]):
                    gp.set_parameter_vector(x[i, :])
                    ll = gp.log_likelihood(F, quiet=True)
                    f[i, :] = -ll if np.isfinite(ll) else 1e25

        if self.opt == "ga":
            X, _, _ = GeneticAlgorithm(
                pop_size=20,
                sampling=LHS().sample_by_bounds(0, 1, n_hyper_var, 100, {}),
                selection=TournamentSelection(),
                crossover=SimulatedBinaryCrossover(),
                mutation=PolynomialMutation(),
                survival=FitnessSurvival(),
                verbose=0
            ).solve(HyperparameterProblem(), 8000)

            gp.set_parameter_vector(X[0, :])

        elif self.opt == "best_lhs":

            n_initial_points = 1000

            p = LHS().sample_by_bounds(0, 1, n_hyper_var, 1000, {})
            likelihoods = np.zeros(n_initial_points)
            for i, row in enumerate(p):
                likelihoods[i] = nll(row)

            print()


        elif self.opt == "lhs":

            initial_points = LHS().sample(HyperparameterProblem(), n_restarts, {})

            likelihoods = np.zeros(n_restarts)
            X = np.zeros((n_restarts, n_hyper_var))
            for i, p in enumerate(initial_points):
                result = op.minimize(nll, p, jac=grad_nll, method="L-BFGS-B")
                likelihoods[i] = result.fun
                X[i, :] = result.x

            idx = np.argmin(likelihoods)
            gp.set_parameter_vector(X[idx, :])

        # p0 = gp.get_parameter_vector()
        # results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
        # gp.set_parameter_vector(results.x)



        self.model = gp
        return self

    @staticmethod
    def get_params():
        val = []
        for kernel in ['linear', 'expsqrt', 'rational_quad', 'exp']:  # , , 'exp', , 'polynomial']:
            val.append({'kernel': kernel, 'opt': 'lhs'})
            #val.append({'kernel': kernel, 'opt': 'ga'})
        return val
