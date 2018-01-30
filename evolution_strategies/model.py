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
        self.n_in = n_in
        self.n_out = n_out
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.dense_in = nn.Linear(n_in, hidden_neurons[0])
        self.hidden = []
        for i in range(hidden_layers):
            exec(f"self.hidden_{i + 1} = nn.Linear(hidden_neurons[{i}], hidden_neurons[{i}])")
            exec(f"self.hidden.append(self.hidden_{i + 1})")
        self.dense_out = nn.Linear(hidden_neurons[-1], n_out)

    def forward(self, x):
        x = F.selu(self.dense_in(x))

        for f in self.hidden:
            x = F.selu(f(x))

        return F.sigmoid(self.dense_out(x))

    def es_params(self):
        """
        The parameters that will be trained by ES.
        :return: (list)
        """
        return [(k, v) for k, v in zip(self.state_dict().keys(),
                                       self.state_dict().values())]


if __name__ == "__main__":
    # test a model
    from torch.autograd import Variable
    m = Model(2, 10, 1)
    print(m)
    x = Variable(torch.FloatTensor([1, 1]))
    print(m(x))