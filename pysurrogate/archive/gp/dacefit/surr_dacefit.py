import matlab.engine
import numpy as np

from pysurrogate.surrogate import Surrogate
from pysurrogate.util.matlabengine import MatlabEngine


class Dacefit(Surrogate):

    def __init__(self, regr, corr, ARD):
        Surrogate.__init__(self)
        self.regr = regr
        self.corr = corr
        self.ARD = ARD
        self.model = None
        self.dmodel = None

    def _predict(self, X):
        mat_F = MatlabEngine.get_instance().predictor(matlab.double(X.tolist()), self.dmodel, nargout=1)
        F = np.array(mat_F._data)
        return F, np.zeros(F.shape)

    def _fit(self, X, F):

        n_var = X.shape[1]

        mat_X = matlab.double(X.tolist())
        mat_F = matlab.double(F.tolist())

        eng = MatlabEngine.get_instance()

        if self.ARD:
            mat_theta0 = eng.transpose(matlab.double([0.1] * n_var))
            mat_theta_L = eng.transpose(matlab.double([0.0001] * n_var))
            mat_theta_U = eng.transpose(matlab.double([20] * n_var))
        else:
            mat_theta0 = matlab.double([0.1])
            mat_theta_L = matlab.double([0.0001])
            mat_theta_U = matlab.double([20])

        mat_regr = matlab.int16([convert_regr(self.regr)])
        mat_corr = matlab.int16([convert_corr(self.corr)])

        self.dmodel = eng.dace_fit(mat_X, mat_F, mat_regr, mat_corr, mat_theta0,
                                                           mat_theta_L, mat_theta_U, nargout=1)

    @staticmethod
    def get_params():
        val = []
        for corr in ['cubic', 'gauss', 'gauss', 'spline']:  # ,  'expg', 'exp']:
            for regr in ['constant', 'linear']:  # , 'quadratic']:
                for ARD in [False, True]:
                    val.append({'corr': corr, 'regr': regr, 'ARD': ARD})
        return val


def convert_regr(str):
    if str == "constant":
        return 0
    elif str == "linear":
        return 1
    elif str == 'quadratic':
        return 2


def convert_corr(str):
    if str == "cubic":
        return 0
    elif str == "exp":
        return 1
    elif str == 'expg':
        return 2
    if str == "gauss":
        return 3
    elif str == "gauss":
        return 4
    elif str == 'spline':
        return 5