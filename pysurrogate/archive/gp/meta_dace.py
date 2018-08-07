import warnings

import numpy as np

from pysurrogate.archive.gp.gpml.gp import GaussianProcess


class DACE(Metamodel):
    def __init__(self, regr, corr, ARD):
        Metamodel.__init__(self)
        self.regr = regr
        self.corr = corr
        self.ARD = ARD
        self.model = None

    def _predict(self, X):
        return self.model.predict(X, eval_MSE=True)

    def _fit(self, X, F, data):
        warnings.filterwarnings('ignore')
        if 'expensive' in data and data['expensive']:
            n_restarts = 20
        else:
            n_restarts = 3

        n_var = X.shape[1]

        if self.ARD:
            theta = (np.full((1, n_var), 0.1), np.full((1, n_var), 1e-5), np.full((1, n_var), 1e5))
        else:
            theta = (0.1, 0.0001, 1000)

        model = GaussianProcess(regr=self.regr,
                                corr=self.corr,
                                random_start=n_restarts,
                                optimizer='fmin_cobyla',
                                #optimizer='f_min',
                                theta0=theta[0],
                                thetaL=theta[1],
                                thetaU=theta[2],
                                nugget=10e-10
                                )
        self.model = model.fit(X, F)
        return self

    @staticmethod
    def get_params():
        val = []
        for corr in ['absolute_exponential', 'squared_exponential', 'cubic', 'linear']:
            for regr in ['constant', 'linear']:# , 'quadratic']:
                val.append({'corr': corr, 'regr': regr, 'ARD': False})
                #val.append({'corr': corr, 'regr': regr, 'ARD': True})
        return val
