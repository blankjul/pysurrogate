import numpy as np
import torch
from torch.autograd import Variable

from pysao.metamodels.metamodel import Metamodel


class Torch(Metamodel):
    def __init__(self):
        Metamodel.__init__(self)
        self.models = None

    def _predict(self, X):
        t_X = Variable(torch.from_numpy(X)).float()
        val = np.zeros((X.shape[0], 1, len(self.models)))
        for i, m in enumerate(self.models):
            val[:, :, i] = m(t_X).data.numpy()
        return np.mean(val, axis=2)[:,0], np.mean(np.std(val, axis=2), axis=0)

    def _fit(self, X, F, data):

        self.models = []

        n_var = X.shape[1]
        if 'expensive' in data and data['expensive']:
            n_metamodels = 1
        else:
            n_metamodels = 1

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

            net = Net(n_feature=n_var, n_hidden=20, n_hidden2=20, n_output=1)  # define the network

            optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
            loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

            for t in range(4 * 50000):
                prediction = net(var_x)  # input x and predict based on x
                loss = loss_func(prediction, var_y)  # must be (1. nn output, 2. target)
                optimizer.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                if t % 10000 == 0:
                    print(loss.data[0])

                if loss.data[0] < 1e-6:
                    break

            print("Finished with error: %s (%s)" % (loss.data[0], t))
            self.models.append(net)


    @staticmethod
    def get_params():
        return [{}]
