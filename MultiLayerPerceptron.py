"""
@auther: Tziporah
"""
import pandas as pd
import warnings

from Perceptron import Perceptron


warnings.simplefilter(action='ignore', category=FutureWarning)


class MLP:
    """A multi-layer Perceptron object with n hidden layers and 1 output layer"""

    def __init__(self, n_hidden: int, weights: list[list], biases: list):
        hidden_layers = [None] * n_hidden
        for i in range(n_hidden):
            hidden_layers[i] = Perceptron(weights[i], biases[i])
        output_layers = [Perceptron(weights[-1], biases[-1])]

        self.n_hidden = n_hidden
        self.n_output = 1
        self.hidden_layers: list[Perceptron] = hidden_layers
        self.output_layers: list[Perceptron] = output_layers
        self.delta_h = [None] * n_hidden
        self.delta_o = [None]

    def _feed_forward(self, inputs: list, desired_output: float):
        h_output = [None] * self.n_hidden

        for i in range(self.n_hidden):
            self.hidden_layers[i].calc_activation(inputs)
            h_output[i] = self.hidden_layers[i].get_activation()

        for i in range(self.n_output):
            self.output_layers[i].calc_activation(h_output)
            o = self.output_layers[i].get_activation()
            self.delta_o[i] = (desired_output - o) * o * (1 - o)
            self.output_layers[i].set_delta(self.delta_o[i])

        return h_output

    def _back_propagate(self, inputs: list, h_output: list, eta: float):
        for i in range(self.n_hidden):
            self.delta_h[i] = (1 - h_output[i]) * h_output[i] * (sum(self.delta_o) * self.output_layers[0].weights[i])

        for i in range(self.n_output):
            self.output_layers[i].set_delta_weights(h_output, eta)
            self.output_layers[i].update_weights()
            self.output_layers[i].update_bias(eta)

        for i in range(self.n_hidden):
            self.hidden_layers[i].set_delta(self.delta_h[i])
            self.hidden_layers[i].set_delta_weights(inputs, eta)
            self.hidden_layers[i].update_weights()
            self.hidden_layers[i].update_bias(eta)

    def ffbp(self, inputs: list, desired_output: float, eta: float, epochs: int = 1):
        for epoch in range(epochs):
            ff = self._feed_forward(inputs, desired_output)
            self._back_propagate(inputs, ff, eta)

    def online_train(self, n_cycles: int, inputs: pd.DataFrame, desired_outputs: pd.Series):
        pass

    def batch_train(self, n_cycles: int):
        pass
