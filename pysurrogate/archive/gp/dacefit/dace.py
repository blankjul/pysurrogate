import numpy as np

from pysurrogate.archive.gp.dacefit.corr import corr_gauss
from pysurrogate.archive.gp.dacefit.regr import regr_constant


class DACE:

    def __init__(self, regr=regr_constant, kernel=corr_gauss):
        super().__init__()
        self.regr = regr
        self.kernel = kernel


    def fit(self, X, Y):

        # the targets should be a 2d array
        if len(Y.shape) == 1:
            Y = Y[:, None]

        # check if for each observation a target values exist
        if X.shape[0] != Y.shape[0]:
            raise Exception("X and Y must have the same number of rows.")

        # save the mean and standard deviation of the input
        mX, sX = np.mean(X, axis=0), np.std(X, axis=0, ddof=1)
        mY, sY = np.mean(Y, axis=0), np.std(Y, axis=0, ddof=1)

        # standardize the input
        nX = (X - mX) / sX
        nY = (Y - mY) / sY

        self.theta = 0.1 * np.ones((X.shape[1], 1))

        # fit the model and set the data
        self.model = fit(nX, nY, self.regr, self.kernel, self.theta)
        self.model = {**self.model, 'mX': mX, 'sX': sX, 'mY': mY, 'sY': sY, 'nX': nX, 'nY': nY}
        self.model['sigma2'] = np.square(sY) @ self.model['_sigma2']

    def predict(self, _X):

        mX, sX, nX = self.model['mX'], self.model['sX'], self.model['nX']
        mY, sY = self.model['mY'], self.model['sY']
        beta, gamma = self.model['beta'], self.model['gamma']

        # normalize the input given the mX and sX that was fitted before
        # NOTE: For the values to predict the _ is added to clarify its not the data fitted before
        _nX = (_X - mX) / sX

        # calculate regression and kernel
        _F = self.regr(_nX)
        _R = calc_kernel_matrix(_nX, nX, self.kernel, self.theta)

        # predict and destandardize
        _sY = _F @ beta + (gamma.T @ _R.T).T
        _Y = (_sY * sY) + mY

        return _Y



if __name__ == "__main__":
    instance = "/Users/julesy/workspace/pysao-benchmark/functions/SL/SL-02-03-01"

    X = np.loadtxt("%s.x_train" % instance)
    Y = np.loadtxt("%s.y_train" % instance)

    Y = np.repeat(Y[:,None], 2, axis=1)
    Y[:,1] = Y[:,1] * 2

    X_test = np.loadtxt("%s.x_test" % instance)

    # kernel = Kernel(SquaredExponential,np.array([0.5] * 10))
    # regr = LinearMean()
    gp = DACE()

    gp.fit(X, Y)
    gp.predict(X_test)
