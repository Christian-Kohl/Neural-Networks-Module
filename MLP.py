import numpy as np


class MLP:
    layers = []
    learning_rate = 1
    classOrReg = False

    def __init__(self, input_size, nodes, output_size, learning_rate):
        np.random.seed(3)
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

    def loss(self, inputs, expected):
        outputs = self.predict(inputs)
        error = 0.5 * np.sum((outputs - expected)**2)
        return error

    def backpropogation(self, inputs, expected):
        outputs = self.feedforward(inputs)
        n_layer = None
        for idx, layer in enumerate(reversed(self.layers)):
            layer.calculate_deltas(expected, outputs, n_layer)
            n_layer = layer
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                layer.update_weights_input_layer(self.learning_rate, inputs)
            else:
                layer.update_weights(self.learning_rate,
                                     self.layers[idx-1])
        return self.layers[0].deltas

    def fit(self, dataset, epochs):
        for epoch in range(0, epochs):
            np.random.shuffle(dataset)
            for point in dataset:
                self.backpropogation(point[:-1], point[-1])


class Layer:
    weights = []
    dotproducts = []
    activations = []
    deltas = []
    last = False
    diffLast = False

    def __init__(self, weights, last=False, diffLast=False):
        self.weights = weights
        self.weights = np.around(self.weights, 1)
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
            return self.activations * (1.0 - self.activations)

        if self.last:
            self.deltas = -(transfer_derivative()) * (expected - results)
        else:
            self.deltas = (transfer_derivative().T) * np.sum(
                           next_layer.weights[:, :-1] * next_layer.deltas.T,
                           axis=0).T

    def update_weights_input_layer(self, learning_rate, inputs):
        input = np.array(([np.append(inputs, [1]), ]*self.deltas.shape[1]))
        multiplier = (-learning_rate*(self.deltas.T*input))
        self.weights = (self.weights + multiplier)

    def update_weights(self, learning_rate, input_layer):
        input = np.array(([np.append(input_layer.activations,
                                     [1]), ]*self.deltas.shape[1]))
        multiplier = (-learning_rate*(self.deltas.T*input))
        self.weights = (self.weights + multiplier)


dataset = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]])
network = MLP(2, [4, 3], 1, 0.1)
network.fit(dataset, 100000)
print(dataset[:, -1])
print(network.predict(dataset[:, :-1]))
