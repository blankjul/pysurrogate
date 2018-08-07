import GPy
import numpy as np

from pysao.metamodels.metamodel import Metamodel


class GPyMetamodel(Metamodel):
    def __init__(self, regr, kernel, ARD):
        Metamodel.__init__(self)
        self.kernel = kernel
        self.ARD = ARD
        self.regr = regr
        self.mean = None
        self.std = None

    def _predict(self, X):
        mean, cov = self.model.predict_noiseless(X)
        mean = mean.T
        cov = cov.T
        if mean.shape[0] == 1:
            mean = mean[0]
        if cov.shape[0] == 1:
            cov = cov[0]

        return mean * self.std + self.mean, cov

    def _fit(self, X, F, data):

        eps = 0.000000001

        n_var = X.shape[1]

        self.mean = np.mean(F)
        self.std = np.std(F)

        # regression type
        if self.regr == 'constant':
            mf = GPy.mappings.Constant(n_var, 1)
        elif self.regr == 'linear':
            mf = GPy.mappings.Linear(n_var, 1)

        # the correlation or kernel
        if self.kernel == "linear":

            if self.ARD:
                variances = np.full(n_var, eps)
            else:
                variances = eps

            kernel = GPy.kern.Linear(n_var, variances=variances, ARD=self.ARD)

        elif self.kernel == "rbf":
            kernel = GPy.kern.RBF(n_var, ARD=self.ARD)
        elif self.kernel == "exp":
            kernel = GPy.kern.Exponential(n_var, ARD=self.ARD)
        elif self.kernel == "expg":
            kernel = GPy.kern.ExpQuad(n_var, ARD=self.ARD)
        else:
            raise ValueError("Parameter %s for kernel unknown." % self.kernel)

        #kernel = kernel + GPy.kern.Linear(X.shape[1], ARD=True)

        F_norm = (F - self.mean) / self.std
        cfl = GPy.models.GPRegression(X, np.array([F_norm]).T, kernel=kernel,
                                      mean_function=mf, noise_var=eps)
        cfl.optimize()
        #cfl.optimize_restarts(num_restarts=4, verbose=False, parallel=True)
        self.model = cfl
        return self

    @staticmethod
    def get_params():
        val = []
        for regr in ['constant', 'linear']:
            for corr in ['rbf', 'expg', 'exp']:
                val.append({'regr': regr, 'kernel': corr, 'ARD': False})
        return val

