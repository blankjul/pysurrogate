from pymop.problem import Problem

from pymoo.algorithms.so_genetic_algorithm import SingleObjectiveGeneticAlgorithm

import numpy as np

def optimize(self):
    n_params = 1

    def evaluate(x, f):
        for i in range(x.shape[0]):
            self.kernel.theta = np.exp(x[i, :] * np.ones(10))
            self._fit()
            f[i, 0] = self.get_neg_log_likelihood()

    class HyperparameterProblem(Problem):
        def __init__(self):
            Problem.__init__(self)
            self.n_var = n_params
            self.n_constr = 0
            self.n_obj = 1
            self.func = self.evaluate_
            self.xl = 0.01 * np.ones(n_params)
            self.xu = 20 * np.ones(n_params)

        def evaluate_(self, x, f):
            evaluate(x, f)

    a = SingleObjectiveGeneticAlgorithm("real", pop_size=50, verbose=False)
    p = HyperparameterProblem()
    [X, _, _] = a.solve(p, 50000)

    self.kernel.theta = np.exp(X[0, :] * np.ones(10))
    self._fit()
