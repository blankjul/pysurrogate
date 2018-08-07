import numpy as np

import matplotlib.pyplot as plt

from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.rand import random
from pymop.griewank import Griewank
from pymop.problem import Problem
from pysamoo.ego.basis import calc_distance_matrix
from pysamoo.ego.distance import eucl_distance, get_parwise


class EvolutionaryInterpolationModel2:

    def __init__(self, xl, xu) -> None:
        super().__init__()

        # the data itself
        self.X = None
        self.Y = None

        # the boundaries
        self.xl = xl
        self.xu = xu

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

    def predict(self, X):
        self.X_weights = np.ones(self.X.shape[0], 1)
        self.eucl_weights = np.ones(self.X.shape[1], 1)

        # calculate the distance matrix
        D = calc_distance_matrix(X, self.X, eucl_distance, xl=self.xl, xu=self.xu, weights=self.weights)

        # calculate the probability of influence for each point
        P = 1 / np.power(D, self.power)

        # if distance was 0 then this point should be dominating all other influences
        P[P == np.inf] = 1e30

        # normalize the used metric to have weights sum up to one
        P /= np.sum(P, axis=1)[:, None]

        # return the interpolated value
        return P @ self.Y

    def optimize(self):
        X = self.X
        n_samples = X.shape[0]
        n_var = X.shape[1]

        class HyperparameterProblem(Problem):
            def __init__(self):
                Problem.__init__(self)
                self.n_var = n_var
                self.n_constr = 0
                self.n_obj = 1
                self.func = self._evaluate
                self.xl = 0 * np.ones(self.n_var)
                self.xu = 1 * np.ones(self.n_var)

            def _evaluate(self, x, f):
                # extract the parameter from the input variable of the hyperparameter problem
                params = x
                n_params = params.shape[0]

                # store each part of the array into the corresponding variable
                eucl_weights = params[:, 0:n_var]

                # get a distance matrix for each parameter setting
                _A, _B = get_parwise(X, X)
                _D = np.repeat((_A - _B)[None, :, :], n_params, axis=0)

                # multiply each distance matrix with euclidean weights and calculate the distance
                _D = np.sum(np.square(_D * eucl_weights[:, None, :]), axis=2)
                _D = np.reshape(_D, (n_params, n_samples, n_samples))

                # calculate for each parameter setting the crossvalidation result
                error = np.full((n_params, n_samples), np.inf)

                _mask = np.full((n_samples, n_samples), True)

                for i in range(n_samples):

                    # mask for crossvalidation calculation
                    mask = np.copy(_mask)
                    mask[i, :] = False
                    mask[: , i] = False

                    # distance matrix for crossvalidation
                    _Dc = _D[:, mask].reshape(n_params,n_samples-1, n_samples-1)

                    pass

                pass

        problem = HyperparameterProblem()
        params = random.random(5, n_var)
        problem.evaluate(params, return_constraints=2)

        pass


if __name__ == '__main__':

    class SimpleProblem(Problem):

        def __init__(self, n_var=1, **kwargs):
            Problem.__init__(self, **kwargs)
            self.n_var = n_var
            self.n_constr = 0
            self.n_obj = 1
            self.func = self._evaluate
            self.xl = 0 * np.ones(self.n_var)
            self.xu = 10 * np.ones(self.n_var)

        def _evaluate(self, x, f):
            f[:, 0] = x[:, 0] * np.sin(x[:, 0])


    # problem = Griewank(n_var=1)
    problem = SimpleProblem()

    X_train = np.sort(RealRandomSampling().sample(problem, 20))
    X_test = np.atleast_2d(np.linspace(problem.xl, problem.xu, 1000)).T

    y_train = problem.evaluate(X_train, return_constraints=0)
    y_test = problem.evaluate(X_test, return_constraints=0)

    model = EvolutionaryInterpolationModel(xl=problem.xl, xu=problem.xu)

    model.fit(X_train, y_train)
    model.optimize()
    y_pred = model.predict(X_test)

    for i in range(y_pred.shape[1]):
        fig = plt.figure()
        plt.plot(X_test, y_test[:, i], label=u'$Problem$')
        plt.plot(X_train, y_train[:, i], 'r.', markersize=10, label=u'Observations')
        plt.plot(X_test, y_pred[:, i], 'b-', label=u'Prediction')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.legend(loc='upper left')
        plt.show()
        plt.close()
