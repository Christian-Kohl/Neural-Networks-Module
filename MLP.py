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
                    self.layers[idx].transfer(input, classOrReg)
                    transfer = self.layers[idx].activations
                else:
                    self.layers[idx].transfer(input, classOrReg)
                    transfer = np.append(self.layers[idx].activations,
                                         np.array([1]))

            else:
                self.layers[idx].transfer(transfer)
                if layer.last:
                    self.layers[idx].transfer(input)
                    transfer = self.layers[idx].activations
                else:
                    self.layers[idx].transfer(input)
                    transfer = np.append(self.layers[idx].activations,
                                         np.array([1]))
        return transfer[0]

    def predict(self, inputs):
        inputs = np.atleast_2d(inputs)
        results = []
        for input in inputs:
            result = self.feedforward(inputs)
            results += [result]
        results = np.array(results)
        return results


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

    def set_deltas(self, expected, results, subsequent_layer=None):

        def transfer_derivative(self):
            return self.dotproducts * (1.0 - self.dotproducts)

        if self.last:
            self.deltas = -(transfer_derivative()) * (expected - results)
        else:
            self.deltas = (transfer_derivative()) * np.sum(
                           subsequent_layer.weights * subsequent_layer.deltas)


dataset = np.array([[2, 1, 2], [4, 2, 8], [3, 3, 9], [5, 10, 50]])
network = MLP(2, [2, 2], 1, 0.8)
print(network.predict(np.array([4, 2])))
