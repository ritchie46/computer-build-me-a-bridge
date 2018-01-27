import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, n_in, n_out, hidden_layers=1, hidden_neurons=(64,)):
        """
        Make a multi-layer-perceptron.

        :param n_in: (int) Input of the model.
        :param n_out: (int) Output of the model.
        :param hidden_layers: (int) No. of hidden layers.
        :param hidden_neurons: (tpl) Hidden neuron per hidden layer.
        """
        super(Model, self).__init__()
        self.dense_in = nn.Linear(n_in, hidden_neurons[0])
        self.hidden = []

        for i in range(hidden_layers):
            self.hidden.append(nn.Linear(hidden_neurons[i], hidden_neurons[i]))
        self.dense_out = nn.Linear(hidden_neurons[-1], n_out)

    def forward(self, x):
        x = F.relu(self.dense_in(x))

        for f in self.hidden:
            x = F.relu(f(x))

        return self.dense_out(x)


if __name__ == "__main__":
    # test a model
    from torch.autograd import Variable
    m = Model(2, 10, 1)
    x = Variable(torch.FloatTensor([1, 1]))
    print(m(x))