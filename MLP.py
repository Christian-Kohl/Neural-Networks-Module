import numpy as np


class MLP:
    weights = []
    learning_rate = 1

    def __init__(self, input_size, nodes, output_size, learning_rate):
        if nodes == []:
            self.weights += [np.random.rand(output_size, input_size+1)]
        else:
            self.weights += [np.random.rand(nodes[0], input_size+1)]
            for layer in range(1, len(nodes)):
                self.weights += [np.random.rand(nodes[layer],
                                 nodes[layer-1] + 1)]
            self.weights += [np.random.rand(output_size, nodes[-1]+1)]
        self.learning_rate = learning_rate


network = MLP(1, [1, 1], 1, 0.8)
print(network.weights)
