pysurrogate - Surrogate Models
==================================

The aim of this project is to build an interface to find the best surrogate for observations made on a blackbox
function. The nature of a blackbox function do not let us make any assumptions about the data and therefore
the model selection needs to be done more carefully.

Installation
==================================

.. code:: bash

    pip install pysurrogate

Surrogates
==================================
Currently the following surrogate models are implemented:

- DACE (Matlab Version, matlab library for python needs to be installed)
- RBF (Scipy)
- NN (pytorch)

Usage
==================================

The method that should be used for interfacing is the pysurrogate.optimize.fit and pysurrogate.optimize.predict
method.


.. code:: python


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
    model = fit(X, Y, n_folds=3, disp=True, normalize_X=True, normalize_Y=True)
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

Contact
==================================
Feel free to contact me if you have any question:

| Julian Blank (blankjul [at] egr.msu.edu)
| Michigan State University
| Computational Optimization and Innovation Laboratory (COIN)
| East Lansing, MI 48824, USA
