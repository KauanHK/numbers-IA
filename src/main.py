def main() -> None:

    import matplotlib.pyplot as plt
    from nn import NeuralNetwork


    nn = NeuralNetwork(learning_rate = 0.01)
    nn.train()

    while True:

        index = int(input("Digite um número (1 - 60000): "))
        img = nn.images[index-1]
        plt.imshow(img.reshape(28, 28), cmap="Greys")

        nn.predict(img)

        plt.title(f"Dígito previsto: {o.argmax()}")
        plt.savefig(f"digit{index}.png")


if __name__ == "__main__":
    main()
