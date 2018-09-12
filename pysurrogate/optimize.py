import time
import warnings

import numpy as np

from pymoo.util.normalization import normalize, denormalize

from pysurrogate.util.error_metrics import calc_mse
from pysurrogate.util.misc import unique_rows, create_crossvalidation_sets


def get_method(name):
    if name == 'scipy_rbf':
        from pysurrogate.surrogates.scipy_rbf import RBF
        return RBF
    elif name == 'matlab_dacefit':
        from pysurrogate.surrogates.matlab_dacefit import Dacefit
        return Dacefit
    elif name == 'sklearn_polyregr':
        from pysurrogate.surrogates.sklearn_polyregr import PolynomialRegression
        return PolynomialRegression
    elif name == 'sklearn_dacefit':
        from pysurrogate.surrogates.sklearn_dacefit import Dacefit
        return Dacefit
    elif name == 'my_dacefit':
        from pysurrogate.surrogates.my_dacefit import MyDacefit
        return MyDacefit
    elif name == 'myrbf':
        from pysurrogate.surrogates.my_rbf import MyRBF
        return MyRBF
    elif name == 'george_gp':
        from pysurrogate.surrogates.gpgeorge_gp import GPGeorge
        return GPGeorge
    elif name == 'gpy_gp':
        from pysurrogate.surrogates.gpy_gp import GPyMetamodel
        return GPyMetamodel
    elif name == 'sklearn_gradient_boosting':
        from pysurrogate.surrogates.sklearn_gradient_boosting import GradientBoosting
        return GradientBoosting
    elif name == 'torch_nn':
        from pysurrogate.surrogates.torch_nn import Torch
        return Torch
    elif name == 'active_subspace_gp':
        from pysurrogate.surrogates.active_subspaces import ActiveSubspacesSurrogate
        return ActiveSubspacesSurrogate
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





def fit(X, Y, methods=['george_gp', 'gpy_gp', 'sklearn_gradient_boosting', 'my_dacefit', 'torch_nn', 'scipy_rbf', 'sklearn_polyregr',
                       'sklearn_dacefit'], func_error=calc_mse, disp=False,
        normalize_X=False, normalize_Y=False,
        do_crossvalidation=True, n_folds=5, crossvalidation_sets=None, debug=False):
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
    debug : bool
        If true warnings and exceptions are shown. Otherwise they are suppressed.
    Returns
    -------

    res : dict
        The res that is used to predict values. It can be heterogenous which means each target value is
        predicted by a different res type with different parameters.

    """

    # if it is only one dimensional convert it
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    if X.shape[0] != Y.shape[0]:
        raise Exception("X and Y does not have the same number of rows!")

    if isinstance(methods, str):
        methods = [methods]

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

        try:
            method, params = get_method_and_params(entry)
        except Exception as e:
            if debug:
                raise e
                warnings.warn(str(e))
                warnings.warn("Not able to load model %s. Will be skipped." % entry)
            continue

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
            for k, entry in enumerate(surrogates):

                try:
                    name, method, param = entry['name'], entry['method'], entry['param']

                    error = np.full(n_folds, np.inf)
                    duration = np.full(n_folds, np.nan)

                    # on each validation set
                    for i, (training, test) in enumerate(crossvalidation_sets):
                        impl = method(**param)

                        start_time = time.time()
                        warnings.filterwarnings("ignore")
                        impl.fit(X[training, :], Y[training, [m]])
                        duration[i] = time.time() - start_time

                        Y_hat = impl.predict(X[test, :], return_std=False)
                        error[i] = func_error(Y[test, [m]], Y_hat)

                except Exception as e:
                    if debug:
                        print(e)
                        warnings.warn("Error while using fitting: %s %s %s" % (name, method, param))

                result.append(
                    {'name': name, 'method': method, 'param': param, 'error': error, 'duration': np.mean(duration)})

            result = sorted(result, key=lambda e: np.mean(e['error']))

            if disp:
                __display(result, str(m+1))

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

    res['surrogates'] = models

    return res


def predict(res, X):
    # if it is only one dimensional convert it
    if X.ndim == 1:
        X = X[:, None]

    Y = np.full((X.shape[0], len(res['surrogates'])), np.inf)

    # denormalize if normalized before
    if res['normalize_X']:
        X = normalize(X, res['X_min'], res['X_max'])

    # for each target value to predict there exists a model
    for m, model in enumerate(res['surrogates']):
        Y[:, m] = model.predict(X)

    # denormalize target if done while fitting
    if res['normalize_Y']:
        Y = denormalize(Y, res['Y_min'], res['Y_max'])

    return Y



def display(model):
    for m, result in enumerate(model['crossvalidation']):
        __display(result,(m+1))


def __display(result, lbl_obj):

    for i in range(len(result)):

        entry = result[i]

        attrs = [('name', entry['name'], 27),
                 ('error', "%.5f" % np.mean(entry['error']), 7),
                 ('duration (sec)', "%.5f" % np.mean(entry['duration']), 7),
                 ('param', entry['param'], 200),
                 ]

        regex = " | ".join(["{}"] * len(attrs))

        if i == 0:
            print("=" * 50)
            print("Target %s" % lbl_obj)
            print("=" * 50)
            print(regex.format(*[name.ljust(width) for name, _, width in attrs]))
            print("-" * 50)

        print(regex.format(*[str(val).ljust(width) for _, val, width in attrs]))

