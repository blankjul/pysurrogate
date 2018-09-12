
import numpy as np
import matplotlib.pyplot as plt
from pysurrogate.optimize import fit, predict


if __name__ == '__main__':

    # number of samples we will use for this example
    n_samples = 20

    # ---------------------------------------------------------
    # Example 1: One input variable and one target
    # ---------------------------------------------------------

    X = np.random.rand(n_samples, 1) * 4 * np.pi
    Y = np.cos(X)

    # fit the model and predict the data
    #model = fit(X, Y, n_folds=3, disp=True, normalize_X=True, normalize_Y=True)

    model = fit(X, Y, methods='my_dacefit', disp=True)

    _X = np.linspace(0, 4 * np.pi, 1000)
    _Y = predict(model, _X)

    plt.scatter(X, Y, label="Observations")
    plt.plot(_X, _Y, label="True")
    plt.show()

    # ---------------------------------------------------------
    # Example 2: Two input variables and two targets.
    #            Normalize before building the model and use only an RBF implementation with a specific kernel
    #            Finally validate the model error on the true function.
    # ---------------------------------------------------------

    X = (np.random.rand(n_samples, 2) * 200) + 500
    func_eval = lambda X: np.concatenate([np.sum(np.square(X), axis=1)[:, None], np.sum(np.sqrt(X), axis=1)[:, None]], axis=1)
    Y = func_eval(X)

    # fit the model and predict the data
    model = fit(X, Y, n_folds=3, disp=True, normalize_X=True, normalize_Y=True)

    # create two dimensional data to test the
    M = np.meshgrid(np.linspace(100, 200, 1000), np.linspace(100, 200, 1000))
    _X = np.concatenate([X[:, :, None] for e in X], axis=2).reshape(n_samples * n_samples, 2)
    _Y = predict(model, _X)
    print(np.mean(np.abs(_Y - func_eval(_X)), axis=0))
