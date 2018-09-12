
import numpy as np
import matplotlib.pyplot as plt
from pysurrogate.optimize import fit, predict


if __name__ == '__main__':

    # number of samples we will use for this example
    n_samples = 100

    # ---------------------------------------------------------
    # Example 1: One input variable and one target
    # ---------------------------------------------------------

    X = np.random.rand(n_samples, 20) * 4 * np.pi
    Y = np.cos(X)

    # fit the model and predict the data
    model = fit(X, Y, methods='sklearn_dacefit', disp=True, debug=True)

    _X = np.linspace(0, 4 * np.pi, 1000)
    _Y = predict(model, _X)

    plt.scatter(X, Y, label="Observations")
    plt.plot(_X, _Y, label="True")
    plt.show()
