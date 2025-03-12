import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from data import get_mnist


def sigmoid(x: int | float):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:

    def __init__(self, learning_rate: float = 0.01) -> None:

        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.uniform(-0.5, 0.5, (16, 784))
        self.weights_hidden_output = np.random.uniform(-0.5, 0.5, (10, 16))
        self.bias_input_hidden = np.zeros((16, 1))
        self.bias_hidden_output = np.zeros((10, 1))

    def forward(self, image: NDArray, label: NDArray) -> tuple[NDArray, NDArray]:

        image = image.reshape(-1, 1)
        label = label.reshape(-1, 1)

        hidden_layer = sigmoid(self.weights_input_hidden @ image + self.bias_input_hidden)
        output = sigmoid(self.weights_hidden_output @ hidden_layer + self.bias_hidden_output)

        return hidden_layer, output

    def backpropagation(self, image: NDArray, label: NDArray, hidden: NDArray, output: NDArray) -> None:

        cost_output = output - label
        self.weights_hidden_output += -self.learning_rate * cost_output @ np.transpose(hidden)
        self.bias_hidden_output += -self.learning_rate * cost_output

        cost_hidden = np.transpose(self.weights_hidden_output) @ cost_output * (hidden * (1 - hidden))
        self.weights_input_hidden += -self.learning_rate * cost_hidden @ np.transpose(image)
        self.bias_input_hidden += -self.learning_rate * cost_hidden

    def train(self) -> None:

        self.images, self.labels = get_mnist()

        i = 1
        for image, label in zip(self.images, self.labels):
            hidden, output = self.forward(image, label)
            self.backpropagation(image, label, hidden, output)
            i += 1
            if i % 1000 == 0:
                print(f'{i:.}/60.000', end = '\r')

    def predict(self, image: NDArray) -> int:

        hidden_layer = sigmoid(self.weights_input_hidden @ image.reshape(784, 1) + self.bias_input_hidden)
        return sigmoid(self.weights_hidden_output @ hidden_layer + self.bias_hidden_output).argmax()
