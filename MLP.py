import numpy as np
from sklearn.utils import shuffle


class MLP:
    layers = []
    learning_rate = 1
    classOrReg = False
    momentum = 0
    reg_term = 0

    def __init__(self,
                 input_size,
                 nodes,
                 output_size,
                 learning_rate,
                 momentum=None,
                 reg_term=0):
        np.random.seed(3)
        self.layers = []

        if nodes == []:
            self.layers += Layer(np.random.rand(output_size, input_size+1),
                                 reg_terms=reg_term)
        else:
            self.layers += [Layer(np.random.rand(nodes[0], input_size+1),
                                  reg_terms=reg_term)]
            for layer in range(1, len(nodes)):
                self.layers += [Layer(np.random.rand(nodes[layer],
                                      nodes[layer-1] + 1))]
            self.layers += [Layer(np.random.rand(output_size, nodes[-1]+1),
                                  last=True,
                                  reg_terms=reg_term)]
        self.learning_rate = learning_rate
        if momentum is None:
            self.momentum = 0
        else:
            self.momentum = momentum
        if reg_term is None:
            self.reg_term = 0
        else:
            self.reg_term = reg_term

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

    def mse_loss(self, inputs, expected):
        if self.reg_term == 0:
            error = self.mse_loss_wo_reg(inputs, expected)
        else:
            error = self.mse_loss_w_reg(inputs, expected)
        return error

    def mse_loss_w_reg(self, inputs, expected):
        outputs = self.predict(inputs)
        acc_term = outputs.shape[0] * np.sum((expected - outputs.T)**2)
        sum = 0
        for layer in self.layers:
            sum += np.sum(np.square(layer.weights))
        regularisation_error = self.reg_term * sum
        error = acc_term + regularisation_error
        return error

    def mse_loss_wo_reg(self, inputs, expected):
        outputs = self.predict(inputs)
        error = outputs.shape[0] * np.sum((expected - outputs.T)**2)
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
                                     self.layers[idx-1],
                                     self.momentum)
        return self.layers[0].deltas

    def fit(self, X_train, y_train, X_test=None, y_test=None, epochs=100):
        if y_train is None:
            for epoch in range(1, epochs+1):
                shuffle(X_train, y_train)
                for idx, point in enumerate(X_train):
                    self.backpropogation(point, y_train[idx])
                print('Epoch number: ', epoch, '  error: ',
                      self.mse_loss(X_train, y_train))
        else:
            log = []
            for epoch in range(1, epochs+1):
                shuffle(X_train, y_train)
                for idx, point in enumerate(X_train):
                    self.backpropogation(point, y_train[idx])
                train_loss = self.mse_loss_wo_reg(X_train,
                                                  y_train)/X_train.shape[1]
                test_loss = self.mse_loss_wo_reg(X_test,
                                                 y_test)/X_test.shape[1]
                log += [[train_loss, test_loss]]
                print('Epoch number: ',
                      epoch,
                      '  error: ',
                      train_loss,
                      'test_error: ',
                      test_loss
                      )
            return log


class Layer:
    weights = []
    dotproducts = []
    activations = []
    deltas = []
    last = False
    diffLast = False
    prev_inc = None
    reg_term = 0

    def __init__(self, weights, last=False, diffLast=False, reg_terms=0):
        self.weights = weights
        self.weights = np.around(self.weights, 1)
        self.last = last
        self.diffLast = diffLast
        self.prev_inc = None
        self.reg_term = reg_terms

    def transfer(self, inputs):
        if self.last:
            self.transfer_lin(inputs)
        else:
            self.transfer_ReLU(inputs)

    def transfer_ReLU(self, inputs):
        dots = np.dot(self.weights, np.atleast_2d(inputs).T)
        self.dotproducts = dots
        act = np.where(self.dotproducts >= 0,
                       self.dotproducts,
                       self.dotproducts*0.01)
        self.activations = np.array(act)

    def transfer_lin(self, inputs):
        dots = np.dot(self.weights, np.atleast_2d(inputs).T)
        self.dotproducts = dots
        act = self.dotproducts
        self.activations = np.array(act)

    def calculate_deltas(self, expected, results, next_layer):

        def transfer_derivative():
            if self.last:
                return 1
            else:
                return np.where(self.dotproducts >= 0,
                                1,
                                0.01)
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

    def update_weights(self, learning_rate, input_layer, momentum):
        input = np.array(([np.append(input_layer.activations,

                                     [1]), ]*self.deltas.T.shape[0]))
        multiplier = -learning_rate*(self.deltas.T*input)
        if self.prev_inc is None:
            inc = multiplier - (self.reg_term*self.weights)
        else:
            reg = - (self.reg_term*self.weights)
            inc = multiplier + (momentum * self.prev_inc) + reg
        self.weights = self.weights + inc
        self.prev_inc = inc
