import numpy as np
import torch
from torch.autograd import Variable

from pysurrogate.surrogate import Surrogate


class Torch(Surrogate):

    def _predict(self, X):
        t_X = Variable(torch.from_numpy(X)).float()
        val = np.zeros((X.shape[0], 1, len(self.model)))
        for i, m in enumerate(self.model):
            val[:, :, i] = m(t_X).data.numpy()
        return np.median(val, axis=2)[:, 0], np.mean(np.std(val, axis=2), axis=0)

    def _fit(self, X, F):

        self.model = []
        n_var = X.shape[1]
        n_metamodels = 1

        F = F[:, None]

        for i in range(n_metamodels):
            var_x, var_y = Variable(torch.from_numpy(X)).float(), Variable(torch.from_numpy(F)).float()

            class Net(torch.nn.Module):
                def __init__(self, n_feature, n_hidden, n_hidden2, n_output):
                    super(Net, self).__init__()
                    self.hidden = torch.nn.Linear(n_feature, n_hidden)
                    self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)
                    self.predict = torch.nn.Linear(n_hidden2, n_output)

                def forward(self, x):
                    x = torch.nn.functional.relu(self.hidden(x))
                    x = torch.nn.functional.relu(self.hidden2(x))
                    x = self.predict(x)
                    return x

            net = Net(n_feature=n_var, n_hidden=20, n_hidden2=20, n_output=1)

            optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
            loss_func = torch.nn.MSELoss()

            for t in range(4 * 50000):
                prediction = net(var_x)
                loss = loss_func(prediction, var_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss.item() < 1e-4:
                    break

            self.model.append(net)
