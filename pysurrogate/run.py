from pysurrogate.optimize import fit, predict

import numpy as np


if __name__ == '__main__':

    X = np.random.rand(40, 3)
    Y = np.random.rand(40,2)
    model = fit(X, Y, n_folds=3, disp=True, normalize_X=True, normalize_Y=True)
    Y_hat = predict(model, X)

    print(Y_hat)
