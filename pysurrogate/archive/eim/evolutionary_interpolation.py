import numpy as np

import matplotlib.pyplot as plt
import scipy

from metamodels.impl.meta_dace import DACE
from metamodels.impl.meta_matlab import Matlab
from metamodels.impl.meta_rbf import RBF
from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.rand import random
from pymop.griewank import Griewank
from pymop.problem import Problem
from pysamoo.ego.basis import calc_distance_matrix
from pysamoo.ego.distance import eucl_distance, get_parwise


class EvolutionaryInterpolationModel:

    def __init__(self, xl, xu, eucl_weights=None) -> None:
        super().__init__()

        # the data itself
        self.X = None
        self.Y = None

        # the boundaries
        self.xl = xl
        self.xu = xu

        self.power = 10
        self.eucl_weights = eucl_weights

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

    def optimize(self):
        n_samples = self.X.shape[0]
        n_var = self.X.shape[1]

        power = np.array([2])
        eucl_weights = np.ones(n_var) / n_var
        x0 = np.concatenate([power, eucl_weights])

        def func(x):
            error = np.full(n_samples, np.inf)

            # do leave one out crossvalidation for each entry
            for i in range(n_samples):

                train = np.array([j for j in range(n_samples) if j != i])
                test = np.array([i])

                obj = EvolutionaryInterpolationModel(xl=self.xl, xu=self.xu, eucl_weights=x[1:])
                obj.power = x[0]
                obj.fit(self.X[train, :], self.Y[train])

                error[i] = np.sum(np.abs(self.Y[test] - obj.predict(self.X[test, :])))

            return np.mean(error)

        xopt = scipy.optimize.fmin(func, x0, disp=True, maxfun=10000)

        self.power = 1
        self.eucl_weights = xopt[1:]


    def predict(self, X):

        # calculate the distance matrix
        D = calc_distance_matrix(X, self.X, eucl_distance, xl=self.xl, xu=self.xu, weights=self.eucl_weights)

        # calculate the probability of influence for each point
        P = 1 / np.power(D, self.power)

        # if distance was 0 then this point should be dominating all other influences
        P[P == np.inf] = 1e30

        # normalize the used metric to have weights sum up to one
        P /= np.sum(P, axis=1)[:, None]

        # return the interpolated value
        return P @ self.Y


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
            f[:, 0] = x[:, 0]
            #f[:, 0] = np.square((x[:, 0] - 5))
            #f[:, 0] = 5 * np.abs(x[:, 0]) * np.sin(x[:, 0])
            #f[:, 0] = np.cos(x[:, 0])


    #problem = Griewank(n_var=1)
    problem = SimpleProblem()

    X_train = np.sort(RealRandomSampling().sample(problem, 20))
    X_test = np.atleast_2d(np.linspace(problem.xl-5, problem.xu+5, 10000)).T

    y_train = problem.evaluate(X_train, return_constraints=0)
    y_test = problem.evaluate(X_test, return_constraints=0)

    model = EvolutionaryInterpolationModel(xl=problem.xl, xu=problem.xu)
    #model = Matlab(regr='constant', corr='gauss', ARD=False)
    #model = DACE(regr='linear', corr='squared_exponential', ARD=False)
    #model = RBF(kernel='multiquadric')

    model.fit(X_train, y_train)
    #model.optimize()
    y_pred = model.predict(X_test)

    y_train_hat = model.predict(X_train)

    if len(y_pred.shape) == 1:
        y_pred = y_pred[:,None]

    print(np.mean(np.abs(y_train - y_train_hat)))

    for i in range(y_pred.shape[1]):
        fig = plt.figure()
        plt.plot(X_train, y_train[:, i], 'r.', markersize=10, label=u'Observations')
        plt.plot(X_test, y_pred[:, i], 'b-', label=u'Prediction')
        plt.plot(X_test, y_test[:, i], 'g:',  label=u'$Problem$')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.legend(loc='upper left')
        plt.show()
        plt.close()
