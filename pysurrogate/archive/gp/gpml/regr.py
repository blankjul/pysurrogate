class MeanFunction:
    def __init__(self, X, beta):
        super().__init__()
        self.X = None
        self.beta = None

    def predict(self, X):
        return X @ self.beta

    def num_of_hyperparameters(self):
        raise ValueError("Must be implemented!")


class ZeroMean(MeanFunction):
    def num_of_hyperparameters(self):
        return 0

class ConstantMean(MeanFunction):
    def num_of_hyperparameters(self):
        return 1

class LinearMean(MeanFunction):
    def num_of_hyperparameters(self):
        return self.X.shape[1]



