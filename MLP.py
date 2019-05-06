import numpy as np


class MLP:
    layers = []
    learning_rate = 1
    classOrReg = False

    def __init__(self, input_size, nodes, output_size, learning_rate):
        np.random.seed(1)
        self.layers = []

        if nodes == []:
            self.layers += Layer(np.random.rand(output_size, input_size+1))
        else:
            self.layers += [Layer(np.random.rand(nodes[0], input_size+1))]
            for layer in range(1, len(nodes)):
                self.layers += [Layer(np.random.rand(nodes[layer],
                                      nodes[layer-1] + 1))]
            self.layers += [Layer(np.random.rand(output_size, nodes[-1]+1),
                                  last=True)]
        self.learning_rate = learning_rate

    def feedforward(self, inputs):
        input = np.append(np.array(inputs), np.array([1]))
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                if layer.last:
                    self.layers[idx].transfer(input)
                    transfer = self.layers[idx].activations
                else:
                    self.layers[idx].transfer(input)
                    transfer = np.append(self.layers[idx].activations,
                                         np.array([1]))
            else:
                if layer.last:
                    self.layers[idx].transfer(transfer)
                    transfer = self.layers[idx].activations
                else:
                    self.layers[idx].transfer(transfer)
                    transfer = np.append(self.layers[idx].activations,
                                         np.array([1]))
        return transfer[0]

    def predict(self, inputs):
        inputs = np.atleast_2d(inputs)
        results = []
        for input in inputs:
            result = self.feedforward(input)
            results += [result]
        results = np.array(results)
        return results

    def backpropogation(self, inputs, expected):
        outputs = self.predict(inputs)
        n_layer = None
        for inp_idx, input in enumerate(inputs):
            for idx, layer in enumerate(reversed(self.layers)):
                layer.calculate_deltas(expected, outputs[inp_idx], n_layer)
                n_layer = self.layers[len(self.layers) - (idx+1)]
            for layer in self.layers:
                layer.update_weights(self.learning_rate)
        return self.layers[0].deltas


class Layer:
    weights = []
    dotproducts = []
    activations = []
    deltas = []
    last = False
    diffLast = False

    def __init__(self, weights, last=False, diffLast=False):
        self.weights = weights
        self.last = last
        self.diffLast = diffLast

    def transfer(self, inputs):
        if self.diffLast:
            self.transfer_last_layer(inputs)
        else:
            self.transfer_other_layer(inputs)

    def transfer_other_layer(self, inputs):
        dots = np.dot(self.weights, np.atleast_2d(inputs).T)
        self.dotproducts = dots
        act = 1 / (1 + np.exp(-self.dotproducts))
        self.activations = np.array(act)

    def transfer_last_layer(self, inputs):
        dots = np.dot(self.weights, np.atleast_2d(inputs).T)
        self.dotproducts = dots
        self.activations = np.array(dots)

    def calculate_deltas(self, expected, results, next_layer):

        def transfer_derivative():
            return self.dotproducts * (1.0 - self.dotproducts)

        if self.last:
            self.deltas = -(transfer_derivative()) * (expected - results)
        else:
            self.deltas = (transfer_derivative()) * np.sum(
                           next_layer.weights * next_layer.deltas)

    def update_weights(self, learning_rate):
        multiplier = (-learning_rate*self.deltas*self.activations)
        self.weights = self.weights + multiplier


dataset = np.array([[2, 1, 2], [4, 2, 8], [3, 3, 9], [5, 10, 50]])
network = MLP(2, [2, 10], 1, 0.8)
print(network.backpropogation(np.array([[2, 1], [4, 2], [1, 1], [3, 3]]),
                              np.array([[2], [8], [1], [9]])))
