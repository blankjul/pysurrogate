import numpy as np
from sklearn.neural_network import MLPRegressor

from pysao.metamodels.metamodel import Metamodel


class NeuralNet(Metamodel):
    def __init__(self, activation, solver, learning_rate):
        Metamodel.__init__(self)
        self.model = None
        self.activation = activation
        self.solver = solver
        self.learning_rate = learning_rate

    def _predict(self, X):
        F = self.model.predict(X)
        return F, np.zeros(X.shape[0])

    def _fit(self, X, F, data):
        clf = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(20,10), max_iter=10000,
                           activation=self.activation, solver=self.solver, learning_rate=self.learning_rate)
        clf.fit(X, F)
        self.model = clf
        return self

    @staticmethod
    def get_params():
        val = []
        for activation in ['identity', 'logistic', 'tanh', 'relu']:
            for solver in ['lbfgs', 'sgd', 'adam']:
                for learning_rate in ['constant', 'invscaling', 'adaptive']:
                    val.append({'activation': activation, 'solver': solver, 'learning_rate': learning_rate})
        return val
