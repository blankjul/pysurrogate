import numpy as np
from scipy.interpolate import Rbf
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

from pysao.metamodels.metamodel import Metamodel


class RBFBoost(Metamodel):
    def __init__(self, rbf_func, alpha, regr):
        Metamodel.__init__(self, normalize=True)
        self.rbf_func = rbf_func
        self.alpha = alpha
        self.model = None
        self.linear = None
        self.regr = regr

        self.beta_norm = None

    def _predict(self, X):

        if self.regr == 'linear':
            X_ = X
        elif self.regr == 'quadratic':
            poly = PolynomialFeatures(degree=2)
            X_ = poly.fit_transform(X)

        # linear prediction
        F = self.linear.predict(X_)[:, 0]

        # RBF error correction
        args = [X[:, i] for i in range(X.shape[1])]
        error = self.model(*args)

        return F + error, np.zeros(X.shape[0])

    def _fit(self, X, F, data):

        if self.regr == 'linear':
            X_ = X
        elif self.regr == 'quadratic':
            poly = PolynomialFeatures(degree=2)
            X_ = poly.fit_transform(X)


        # fit linear first
        self.linear = Ridge(alpha=self.alpha, fit_intercept=True, copy_X=True, normalize=True)
        self.linear.fit(X_, F[:,None])
        error = F - self.linear.predict(X_)[:, 0]

        # fit the error by rbf
        args = [X[:, i] for i in range(X.shape[1])]
        args.append(error)
        self.model = Rbf(*args, epsilon=None, function=self.rbf_func)

        return self

    @staticmethod
    def get_params():
        val = []
        for regr in ['linear', 'quadratic']:
            for rbf in ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate']:
                for alpha in np.linspace(0.01, 10, 20):
                    val.append({'rbf_func': rbf, 'alpha': alpha,  'regr': regr})
        return val

