import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from pysao.metamodels.metamodel import Metamodel


class Keras(Metamodel):
    def __init__(self):
        Metamodel.__init__(self)
        self.model = None

    def _predict(self, X):
        F = self.model.predict(X)
        return F, np.zeros(X.shape[0])

    def _fit(self, X, F, data):
        def get_model():
            model = Sequential()
            model.add(Dense(50, input_dim=X.shape[1], kernel_initializer='uniform', activation='linear'))
            model.add(Dense(20, input_dim=50, activation='relu'))
            model.add(Dense(1, input_dim=20, activation='linear'))
            model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mean_squared_error'])
            return model

        estimator = KerasRegressor(build_fn=get_model, epochs=15000, batch_size=X.shape[0], verbose=2)
        estimator.fit(X, F)

        self.model = estimator
        return self


    @staticmethod
    def get_params():
        return [{}]

