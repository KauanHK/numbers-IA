import numpy as np
from numpy.typing import NDArray
from data import get_mnist


def sigmoid(x: int | float):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:

    def __init__(self, learning_rate: float = 0.01) -> None:
        """Rede neural para identificar o dígito na imagem
        
        :param learning_rate: Taxa de aprendizado, default 0.01
        """

        self._learning_rate = learning_rate
    
        # Inicialização do pesos e bias como matrizes
        # Cada linha representa os pesos de um neurônio
        self._weights_hidden = np.random.uniform(-0.5, 0.5, (16, 784))
        self._weights_output = np.random.uniform(-0.5, 0.5, (10, 16))
        self._bias_hidden = np.zeros((16, 1))
        self._bias_output = np.zeros((10, 1))

        # Imagens e labels do MNIST
        self._images = None
        self._labels = None

    @property
    def images(self) -> NDArray:

        if self._images is None:
            self._images, self._labels = get_mnist()
        return self._images
    
    @property
    def labels(self) -> NDArray:

        if self._labels is None:
            self._images, self._labels = get_mnist()
        return self._labels

    def forward(self, image: NDArray) -> tuple[NDArray, NDArray]:
        """Propagação da imagem pela rede neural. Retorna um tupla com 
        os valores da camada oculta e da camada de saída.
        
        :param image: NDArray da imagem do MNIST (784, 1)
        :returns: hidden_layer, output
        """

        hidden_layer = sigmoid(self._weights_hidden @ image + self._bias_hidden)
        output = sigmoid(self._weights_output @ hidden_layer + self._bias_output)

        return hidden_layer, output

    def backpropagation(self, image: NDArray, label: NDArray, hidden: NDArray, output: NDArray) -> None:
        """Algoritmo de backpropagation para ajustar os pesos e bias da rede neural.
        
        :param image: NDArray da imagem do MNIST (784, 1)
        :param label: NDArray do label da imagem do MNIST (10, 1)
        :param hidden: NDArray da camada oculta (16, 1)
        :param output: NDArray da camada de saída (10, 1)
        """

        # Cálculo do erro, resulta em uma matriz (10, 1)
        cost_output = output - label

        # Atualização dos pesos e bias dos outputs
        # Produto de matrizes (10, 1) @ (1, 16) = (10, 16)
        self._weights_output += -self._learning_rate * cost_output @ np.transpose(hidden)
        self._bias_output += -self._learning_rate * cost_output

        # Erro da camada oculta
        # (784, 16) @ (10, 1) x (16, 1) = (16, 1)
        cost_hidden = np.transpose(self._weights_output) @ cost_output * (hidden * (1 - hidden))
        self._weights_hidden += -self._learning_rate * cost_hidden @ np.transpose(image)
        self._bias_hidden += -self._learning_rate * cost_hidden

    def train(self) -> None:
        """Treina a rede neural com as imagens do MNIST e retorna None."""

        i = 1
        for image, label in zip(self.images, self.labels):

            image = image.reshape(-1, 1)
            label = label.reshape(-1, 1)

            hidden, output = self.forward(image, label)
            self.backpropagation(image, label, hidden, output)
            i += 1
            if i % 1000 == 0:
                print(f'{i}/60.000', end = '\r')

    def predict(self, image: NDArray) -> int:
        """Retorna o dígito previsto na imagem.
        
        :param image: NDArray da imagem do MNIST (784, 1)
        """

        hidden_layer = sigmoid(self._weights_hidden @ image.reshape(784, 1) + self._bias_hidden)
        return sigmoid(self._weights_output @ hidden_layer + self._bias_output).argmax()
