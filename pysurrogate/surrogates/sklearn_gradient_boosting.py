import numpy as np
from sklearn import ensemble

from pysurrogate.surrogate import Surrogate


class GradientBoosting(Surrogate):
    def __init__(self, learning_rate, loss, n_estimators):
        Surrogate.__init__(self)
        self.learning_rate = learning_rate
        self.loss = loss
        self.n_estimators = n_estimators


    def _predict(self, X):
        F = self.model.predict(X)
        return F, np.zeros(X.shape[0])

    def _fit(self, X, F):
        clf = ensemble.GradientBoostingRegressor(
            loss=self.loss,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=1.0,
            criterion='friedman_mse',
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.,
            max_depth=3,
            min_impurity_decrease=0.,
            min_impurity_split=None,
            init=None,
            random_state=None,
            max_features=None,
            alpha=0.9,
            verbose=0,
            max_leaf_nodes=None,
            warm_start=False,
            presort='auto'
        )
        clf.fit(X, F)
        self.model = clf
        return self

    @staticmethod
    def get_params():
        val = []
        for learning_rate in [0.05, 0.1]:
            for loss in ['ls', 'lad', 'huber', 'quantile']:
                for n_estimators in [100]:
                    val.append({'learning_rate': learning_rate, 'loss': loss, 'n_estimators': n_estimators})
        return val


