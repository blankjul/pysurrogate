import gpflow
import numpy
import scipy
from gpflow.mean_functions import Linear, Constant


from pysao.metamodels.metamodel import Metamodel
from pysao.operators.sampling.latin_hypercube import LHS
from pymop.problem import Problem


class GPFlow(Metamodel):
    def __init__(self, regr, kernel, ARD):
        Metamodel.__init__(self)
        self.regr = regr
        self.kernel = kernel
        self.ARD = ARD
        self.model = None

    def _predict(self, X):
        args = [X[:, i] for i in range(X.shape[1])]
        return self.model(*args), None

    def _fit(self, X, F, data):

        if self.regr == 'constant':
            mf = Constant()
        elif self.regr == 'linear':
            mf = Linear(numpy.ones((X.shape[1], 1)), numpy.ones((1, 1)))

        if self.kernel == 'linear':
            kernel = gpflow.kernels.Linear(X.shape[1], ARD=self.ARD)
        if self.kernel == 'rbf':
            kernel = gpflow.kernels.RBF(X.shape[1], ARD=self.ARD)
        elif self.kernel == 'polynomial':
            kernel = gpflow.kernels.Polynomial(X.shape[1], ARD=self.ARD)

        m = gpflow.gpr.GPR(X, numpy.array([F]).T, kern=kernel, mean_function=mf)
        m.optimize()
        self.model = m

    def _predict(self, X):
        mean, cov = self.model.predict_y(X)

        mean = mean.T
        cov = cov.T
        if mean.shape[0] == 1:
            mean = mean[0]
        if cov.shape[0] == 1:
            cov = cov[0]
        return mean, cov

    @staticmethod
    def get_params():
        val = []
        for regr in ['constant', 'linear']:
            for kernel in ['linear', 'rbf', 'polynomial']:
                val.append({'regr': regr, 'kernel': kernel, 'ARD': False})
        return [{'regr': 'constant', 'kernel': 'linear', 'ARD': False}]
        #return val


def minimize(fun, x0):
    n_var = len(x0)
    p = Problem(n_var=n_var, n_obj=1, n_constr=0, xl=-10, xu=10, func=None)

    def evaluate(x):
        f = numpy.zeros((x.shape[0], 1))
        g = numpy.zeros((x.shape[0], 0))

        if x.ndim == 1:
            x = numpy.array([x])
        for i in range(x.shape[0]):
            f[i, :] = fun(x[i, :])[0]
        return f, g

    p.evaluate = evaluate

    # X,F,G = NSGA(pop_size=300).solve(p, 60000)

    # X, F, G = solve_by_de(p)

    print(X)
    print(F)

    return X[0, :]


def minimize2(fun, x0):
    n_var = len(x0)
    val = scipy.optimize.differential_evolution(lambda x: fun(x)[0], [(-10, 10) for _ in range(n_var)],
                                                popsize=40, mutation=(0.7, 1), recombination=0.3, )
    print(val.fun)
    print(val.x)

    return val.x
    # p = Problem(n_var=n_var, n_obj=1, n_constr=0, xl=-10, xu=10, func=None)


def minimize3(fun, x0):
    n_var = len(x0)
    n_restarts = 30

    initial_points = LHS().sample(n_restarts, -10 * numpy.ones(n_var), 10 * numpy.ones(n_var))

    # initial__point = [x0]
    # for _ in range(n_restarts-1):
    #    initial__point.append(numpy.random.random(n_var))

    results = []
    for p in initial_points:
        result = scipy.optimize.minimize(fun=fun,
                                         x0=p,
                                         method='L-BFGS-B',
                                         jac=True,
                                         tol=None,
                                         callback=None)
        results.append(result)

    idx = numpy.argmin([e.fun for e in results])
    result = results[idx]
    print(result.x)
    print(result.fun)
    return result.x
