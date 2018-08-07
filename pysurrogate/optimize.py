import numpy as np

from pymoo.util.normalization import normalize, denormalize

from pysurrogate.util.error_metrics import calc_mse
from pysurrogate.util.misc import unique_rows, create_crossvalidation_sets


def get_method(name):
    if name == 'scipy_rbf':
        from pysurrogate.models.scipy_rbf import RBF
        return RBF
    elif name == 'matlab_dacefit':
        from pysurrogate.models.matlab_dacefit import Dacefit
        return Dacefit
    elif name == 'sklearn_polyregr':
        from pysurrogate.models.sklearn_polyregr import PolynomialRegression
        return PolynomialRegression
    elif name == 'sklearn_dacefit':
        from pysurrogate.models.sklearn_dacefit import Dacefit
        return Dacefit
    elif name == 'myrbf':
        from pysurrogate.models.my_rbf import MyRBF
        return MyRBF
    elif name == 'nn':
        from pysurrogate.models.torch_nn import Torch
        return Torch
    else:
        raise Exception('Surrogate is unknown: %s' % name)


def get_method_and_params(entry):
    # parse the archive and parameters from the input
    if isinstance(entry, str):
        clazz = get_method(entry)
        return clazz, clazz.get_params()
    elif isinstance(entry, tuple):
        clazz = get_method(entry[0])
        return clazz, entry[1]
    else:
        raise Exception(
            "Either a list of strings (each is a model), or a list of tuples (1) model (2) params.")


def fit(X, Y, methods=['nn', 'scipy_rbf', 'sklearn_polyregr', 'sklearn_dacefit'], func_error=calc_mse, disp=False,
        normalize_X=False, normalize_Y=False,
        do_crossvalidation=True, n_folds=10, crossvalidation_sets=None):
    """

    The is the public interface which fits a surrogate res that is able to predict more than one target value.

    Parameters
    ----------
    X : numpy.array
        Design space which is a two dimensional array nxm array, where n is the number of samples and
        m the number of variables.
    Y : numpy.array
        The target values that should be predicted by the res
    methods : list of strings
        A list methods as string which should be considered during the fitting
    func_error : function
        The error metric which is used to compare the surrogate goodness. error(F, F_hat) where it compares the
        prediction F_hat of the res with the true values F.
    disp : bool
        Print output during the fitting of the surrogate archive with information about the error.

    Returns
    -------

    res : dict
        The res that is used to predict values. It can be heterogenous which means each target value is
        predicted by a different res type with different parameters.

    """

    if X.shape[0] != Y.shape[0]:
        raise Exception("X and Y does not have the same number of rows!")

    # the object that is returned in the end having all the necessary information for the prediction
    res = {'n_samples': X.shape[0], 'n_var': X.shape[1], 'n_targets': Y.shape[1],
           'normalize_X': normalize_X, 'normalize_Y': normalize_Y}

    # remove duplicated rows if they occur in the input
    I = unique_rows(X)
    X, Y = X[I, :], Y[I, :]

    # normalize input or target if boolean values set to true
    if normalize_X:
        X, res['X_min'], res['X_max'] = normalize(X, return_bounds=True)
    if normalize_Y:
        Y, res['Y_min'], res['Y_max'] = normalize(Y, return_bounds=True)

    # create a list of all entries that should be run
    surrogates = []
    for entry in methods:
        method, params = get_method_and_params(entry)
        for param in params:
            surrogates.append({'name': entry, 'method': method, 'param': param, 'error': None})

    # list of crossvalidation results - for each target one entry
    crossvalidation = []

    # if the archive should be evaluated using crossvalidation
    if do_crossvalidation:

        # create the sets - either provided or randomly
        if crossvalidation_sets is None and n_folds is not None:
            crossvalidation_sets = create_crossvalidation_sets(res['n_samples'], n_folds, randomize=True)
        if crossvalidation_sets is None:
            raise Exception("Either specify the number of folds or directly provide the crossvalidation sets!")

        for m in range(res['n_targets']):

            # the crossvalidation results are saved in this dictionary - each entry one parameter configuration
            result = []

            # for each method validate
            for entry in surrogates:
                name, method, param = entry['name'], entry['method'], entry['param']

                error = np.full(n_folds, np.inf)

                # on each validation set
                for i, (training, test) in enumerate(crossvalidation_sets):
                    impl = method(**param)
                    impl.fit(X[training, :], Y[training, [m]])

                    Y_hat = impl.predict(X[test, :], return_std=False)
                    error[i] = func_error(Y[test, [m]], Y_hat)

                result.append({'name': name, 'method': method, 'param': param, 'error': error})

            result = sorted(result, key=lambda e: np.mean(e['error']))

            if disp:
                print("Target %s" % (m + 1))
                for i in range(len(result)):
                    entry = result[i]
                    print(entry['name'], entry['param'], entry['error'], np.mean(entry['error']))
                print("=" * 40)

            crossvalidation.append(result)

        res['crossvalidation'] = crossvalidation

    # if no crossvalidation should be done than there is only one res to select
    else:
        if len(surrogates) != 1:
            raise Exception("Please provide exactly one surrogate if no surrogate selection is performed.")
        # add dummy entries here
        for m in range(res['n_targets']):
            crossvalidation.append(surrogates[0])

    # finally fit the res on all available data
    models = []
    for m in range(res['n_targets']):
        # select the best available res found through crossvalidation
        method, param = crossvalidation[m][0]['method'], crossvalidation[m][0]['param']
        impl = method(**param)
        impl.fit(X, Y[:, m])
        models.append(impl)

    res['models'] = models

    return res


def predict(res, X):
    Y = np.full((X.shape[0], len(res['models'])), np.inf)

    # denormalize if normalized before
    if res['normalize_X']:
        X = normalize(X, res['X_min'], res['X_max'])

    # for each target value to predict there exists a model
    for m, model in enumerate(res['models']):
        Y[:, m] = model.predict(X)

    # denormalize target if done while fitting
    if res['normalize_Y']:
        Y = denormalize(Y, res['Y_min'], res['Y_max'])

    return Y
