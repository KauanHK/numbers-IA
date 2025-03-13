def main() -> None:

    import matplotlib.pyplot as plt
    from nn import NeuralNetwork

    nn = NeuralNetwork(learning_rate = 0.01)
    nn.train(3)

    while True:

        try:
            index = int(input("Digite um número (1 - 60000): "))
        except (KeyboardInterrupt, ValueError, TypeError):
            return
        
        img = nn.images[index-1]
        label = nn.labels[index-1]
        plt.imshow(img.reshape(28, 28), cmap="Greys")

        plt.title(f'Dígito previsto: {nn.predict(img)}, Dígito real: {label.argmax()}')
        plt.show()


if __name__ == '__main__':
    main()
