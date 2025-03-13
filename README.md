# Classificador de Dígitos MNIST com Rede Neural

Uma implementação simples de rede neural para classificar dígitos manuscritos do conjunto de dados MNIST.

## Visão Geral

Este projeto implementa uma rede neural feedforward com uma camada oculta para reconhecer dígitos manuscritos (0-9) do conjunto de dados MNIST. A arquitetura da rede consiste em:

- Camada de entrada: 784 neurônios (imagens de 28x28 = 784 pixels)
- Camada oculta: 16 neurônios com ativação sigmoid
- Camada de saída: 10 nós com ativação sigmoid (um para cada dígito)

## Características

- Implementação simples usando apenas NumPy
- Propagação (forward) para previsões
- Backpropagation para treinamento
- Integração com o conjunto de dados MNIST

## Instalação

Você deve ter python instalado, e os módulos numpy e matplotlib:
```bash
pip install numpy matplotlib
```
Clone o repositório:
```bash
git clone https://github.com/KauanHK/numbers-IA
```

## Uso

Para executar, apenas execute:
```bash
python src/main.py
```
Você pode testar outros valores de `learning_rate` ou `epochs`. Basta alterar as seguintes linhas em src/main.py:
```python
nn = NeuralNetwork(learning_rate=0.005)
nn.train(epochs=5)
```

Durante o treinamento, o console exibirá a cada 1000 imagens a porcentagem de precisão atual.

## Detalhes de Implementação

- A rede usa funções de ativação sigmoid para as camadas oculta e de saída
- Os pesos são inicializados aleatoriamente no intervalo [-0.5, 0.5]
- Os bias são inicializados em zero
- A rede carrega automaticamente os dados MNIST
- O algoritmo backpropagation implementa o gradiente descendente para minimizar o erro de previsão

## Desempenho

Com as configurações padrão, a rede pode alcançar uma precisão acima de 93% após algumas épocas de treinamento.
