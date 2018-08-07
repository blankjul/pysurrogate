import numpy as np
import xgboost

from pysao.metamodels.metamodel import Metamodel


class XGBoost(Metamodel):
    def __init__(self):
        Metamodel.__init__(self)

    def _predict(self, X):
        preds = self.model.predict(X)
        return preds, np.zeros(X.shape[0])

    def _fit(self, X, F, data):

        xgb = xgboost.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True,
                                   objective='reg:linear', booster='gbtree', n_jobs=1, nthread=None,
                                   gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                                   colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5,
                                   random_state=0, seed=None, missing=None)
        bst = xgb.fit(X, F)
        self.model = bst
        return self

    @staticmethod
    def get_params():
        val = [{}]
        return val

