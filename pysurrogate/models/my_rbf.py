import copy

import numpy as np

import pymoo
from pysurrogate.surrogate import Surrogate
from pysurrogate.util.distance import calc_distance_matrix, eucl_distance




class MyRBF(Surrogate):

    def __init__(self, basis, optimize, theta=None) -> None:
        super().__init__()

        # the basis function to calculate the kernel
        self.basis = basis
        self.optimize = optimize
        self.theta = theta

        # data itself
        self.X = None
        self.y = None
        self.n_var = None

        # kernel matrix
        self.K = None

        # fitted weights
        self.w = None

        # the actual hyperparameter
        self.theta = None

    def _fit(self, X, y):

        # fit the values to predict
        self.X = X
        self.y = y
        self.n_var = X.shape[1]

        self.func_basis, theta_length = get_basis_function(self.basis, self.n_var)

        # if a hyperparameter is needed at all
        if theta_length is not None:

            if self.optimize is None:

                # create the kernel matrix
                self.K = self.basis.calc(X, X)

                # find the weights according to kernel and target
                self.w = np.linalg.solve(self.K, y)

            elif self.optimize == 'simple':

                h = np.mean(calc_distance_matrix(X, X, eucl_distance))

                self.basis.set_hyperparameter(h)
                self.refit()

            elif self.optimize == 'scipy':
                X_max = np.max(self.X, axis=0)
                X_min = np.min(self.X, axis=0)
                edges = X_max - X_min
                edges = edges[np.nonzero(edges)]
                self.theta = np.power(np.prod(edges) / X.shape[0], 1.0 / edges.size)

            elif optimize == 'ga':
                optimize(self, 5, 0.001, 20)

            else:
                raise Exception("Unknown optimization method: %s" % optimize)

    def refit(self, **kwargs):
        self.fit(self.X, self.y, **kwargs)

    def _predict(self, X):
        D = self.func_basis(X, self.X, self.theta)
        Y = (self.w.T @ D.T).T
        return Y, None

    @staticmethod
    def get_params():
        val = []
        for basis in ['gaussian']:
            for optimize in ['scipy']:
                val.append({'basis': basis, 'optimize': optimize})
        return val


def optimize(model, n_folds, xl, xu):
    def func_evaluate(x):

        f = np.full((x.shape[0], 1), np.inf)

        # the data to use for crossvalidation
        X = model.X
        y = model.y

        # a copy of the original model to optimize on
        m = copy.deepcopy(model)
        error = np.zeros((x.shape[0], len(n_folds)))

        for i in range(x.shape[0]):

            m.basis.set_hyperparameter(np.exp(x[i, :]))

            for j, (train, test) in enumerate(n_folds):
                m.fit(X[train, :], y[train])
                y_pred = m.predict(X[test, :])

                error[i, j] = np.mean(np.square(y[test] - y_pred))

        f[:, 0] = np.mean(error, axis=1)

    res = pymoo.optimize.minimize(func_evaluate, method="ga", xl=xl, xu=xu, n_var=model.n_var)
    model.basis.set_hyperparameter(np.exp(res.X[0, :]))
    model.refit()


def get_basis_function(name, n_var):
    if name == 'gaussian':
        return func_gaussian, n_var
    else:
        raise Exception("Basis not known.")


def func_gaussian(A, B, theta):
    D = calc_distance_matrix(A, B, func=lambda _A, _B: np.sqrt(np.square((_A - _B) * theta)))
    K = np.exp(-np.square(D))
    return K
