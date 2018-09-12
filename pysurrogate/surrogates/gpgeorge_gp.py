import george
import numpy as np
import scipy.optimize as op

from pysurrogate.surrogate import Surrogate


class GPGeorge(Surrogate):
    def __init__(self, kernel):
        Surrogate.__init__(self)
        self.kernel = kernel
        self.model = None
        self.F = None

    def _predict(self, X):
        return self.model.predict(self.F, X, return_var=True)

    def _fit(self, X, F):

        self.F = F
        n_var = X.shape[1]

        if self.kernel == "linear":
            kernel = george.kernels.LinearKernel(order=2, log_gamma2=0.2, ndim=n_var)
        elif self.kernel == "expsqrt":
            kernel = george.kernels.ExpSquaredKernel(metric=np.ones(n_var), ndim=n_var)
        elif self.kernel == "rational_quad":
            kernel = george.kernels.RationalQuadraticKernel(log_alpha=0.2, metric=np.ones(n_var), ndim=n_var)
        elif self.kernel == "exp":
            kernel = george.kernels.ExpKernel(metric=np.ones(n_var), ndim=n_var)
        elif self.kernel == "polynomial":
            kernel = george.kernels.PolynomialKernel(metric=np.ones(n_var))
        else:
            raise ValueError("Parameter %s for kernel unknown." % self.kernel)

        gp = george.GP(kernel, fit_mean=True)

        t = 0.1 * np.ones((n_var, 1))

        # Define the objective function (negative log-likelihood in this case).
        def nll(p):
            gp.set_parameter_vector(p)
            ll = gp.log_likelihood(y, quiet=True)
            return -ll if np.isfinite(ll) else 1e25

        # And the gradient of the objective function.
        def grad_nll(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(y, quiet=True)

        # You need to compute the GP once before starting the optimization.
        gp.compute(t)

        # Print the initial ln-likelihood.
        print(gp.log_likelihood(F))

        # Run the optimization routine.
        p0 = gp.get_parameter_vector()
        results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")

        # Update the kernel and print the final log-likelihood.
        gp.set_parameter_vector(results.x)
        print(gp.log_likelihood(F))

        gp.optimize(X, F)
        self.model = gp


    @staticmethod
    def get_params():
        val = []
        for kernel in ['linear', 'expsqrt', 'rational_quad', 'exp']:  # , , 'exp', , 'polynomial']:
            val.append({'kernel': kernel})
        return val
