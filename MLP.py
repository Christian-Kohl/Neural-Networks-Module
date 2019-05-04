import numpy as np


class MLP:
    weights = []
    learning_rate = 1

    def __init__(self, input_size, nodes, output_size, learning_rate):
        if nodes == []:
            self.weights += Layer([np.random.rand(output_size, input_size+1)])
        else:
            self.weights += Layer([np.random.rand(nodes[0], input_size+1)])
            for layer in range(1, len(nodes)):
                self.weights += Layer([np.random.rand(nodes[layer],
                                      nodes[layer-1] + 1)])
            self.weights += Layer([np.random.rand(output_size, nodes[-1]+1)],
                                  last=True)
        self.learning_rate = learning_rate


class Layer:
    weights = []
    dotproducts = []
    activations = []
    deltas = []
    last = False

    def __init__(self, weights, last=False):
        self.weights = weights
        self.last = last

    def transfer(self, inputs):
        dots = np.dot(self.weights, inputs)
        self.dotproducts = dots
        act = 1 / (1 + np.exp(-self.dotproduct))
        self.transfer = act
        return self.transfer

    def set_deltas(self, expected, results, subsequent_layer=None):

        def transfer_derivative(self):
            return self.dotproducts * (1.0 - self.dotproducts)

        if self.last:
            self.deltas = -(transfer_derivative()) * (expected - results)
        else:
            self.deltas = (transfer_derivative()) * np.sum(
                           subsequent_layer.weights * subsequent_layer.deltas)


network = MLP(1, [1, 1], 1, 0.8)
print(network.weights)
