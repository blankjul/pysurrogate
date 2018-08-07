from abc import abstractmethod


class Surrogate:

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = None

    def fit(self, X, Y):
        self._fit(X, Y)
        return self.model

    def predict(self, X, return_std=False):
        Y_hat, Y_std = self._predict(X)
        if return_std:
            return Y_hat, Y_std
        else:
            return Y_hat

    @abstractmethod
    def _fit(self, X, Y):
        pass

    @abstractmethod
    def _predict(self, Y):
        pass

    @staticmethod
    def get_params():
        return [{}]
