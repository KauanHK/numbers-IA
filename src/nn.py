import numpy as np
import matplotlib.pyplot as plt
from data import get_mnist


def sigmoid(x: int | float) -> float:
    return 1 / (1 + np.exp(-x))


weights_input_hidden = np.random.uniform(-0.5, 0.5, (16, 784))
weights_hidden_output = np.random.uniform(-0.5, 0.5, (10, 16))
bias_input_hidden = np.zeros((16, 1))
bias_hidden_output = np.zeros((10, 1))

images, labels = get_mnist()

LEARNING_RATE = 0.01
i = 0
for image, label in zip(images, labels):

    image = image.reshape(-1, 1)
    label = label.reshape(-1, 1)

    print(i, end = '\r')
    i += 1

    # Forward
    hidden_layer = sigmoid(weights_input_hidden @ image + bias_input_hidden)
    output = sigmoid(weights_hidden_output @ hidden_layer + bias_hidden_output)

    # Backpropagation
    cost_output = output - label
    weights_hidden_output += -LEARNING_RATE * cost_output @ np.transpose(hidden_layer)
    bias_hidden_output += -LEARNING_RATE * cost_output

    cost_hidden = np.transpose(weights_hidden_output) @ cost_output * (hidden_layer * (1 - hidden_layer))
    weights_input_hidden += -LEARNING_RATE * cost_hidden @ np.transpose(image)
    bias_input_hidden += -LEARNING_RATE * cost_hidden


while True:

    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = bias_input_hidden + weights_input_hidden @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = bias_hidden_output + weights_hidden_output @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"Dígito previsto: {o.argmax()}")
    plt.savefig(f"digit{index}.png")
