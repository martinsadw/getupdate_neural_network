import sklearn.datasets

import numpy as np


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def deriv_sigmoid(X):
    return sigmoid(X) * (1 - sigmoid(X))


def mse(T, Y):
    return ((T - Y)**2).mean(axis=1)


def deriv_mse(T, Y):
    return 2 * (T - Y)


def run(network_input, network_output, batch_size, learning_rate, training_epochs):
    (quant_input, input_size) = network_input.shape
    (_, output_size) = network_output.shape

    total_batch = quant_input // batch_size

    layer_size_1 = 10
    layer_size_2 = 10

    w1 = 2 * np.random.rand(input_size, layer_size_1) - 1
    w2 = 2 * np.random.rand(layer_size_1, layer_size_2) - 1
    w3 = 2 * np.random.rand(layer_size_2, output_size) - 1

    for epoch in range(training_epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_input = network_input[i * batch_size:(i + 1) * batch_size]
            batch_output = network_output[i * batch_size:(i + 1) * batch_size]

            l0 = batch_input

            z1 = np.dot(l0, w1)
            l1 = sigmoid(z1)

            z2 = np.dot(l1, w2)
            l2 = sigmoid(z2)

            z3 = np.dot(l2, w3)
            output = sigmoid(z3)

            l3_delta = learning_rate * deriv_mse(batch_output, output) * deriv_sigmoid(z3)
            l2_delta = np.dot(l3_delta, w3.T) * deriv_sigmoid(z2)
            l1_delta = np.dot(l2_delta, w2.T) * deriv_sigmoid(z1)

            w3 += np.dot(l2.T, l3_delta)
            w2 += np.dot(l1.T, l2_delta)
            w1 += np.dot(l0.T, l1_delta)

            avg_cost += mse(batch_output, output).mean() / total_batch

        print("Epoch: {:04d} cost={:.9f}".format((epoch+1), avg_cost))

    return (w1, w2, w3)


# Código principal
if __name__ == "__main__":
    #QUANT_INPUT = 20
    #
    #INPUT_SIZE = 5
    #network_input = np.ones((QUANT_INPUT, INPUT_SIZE))
    #
    #OUTPUT_SIZE = 3
    #network_output = np.ones((QUANT_INPUT, OUTPUT_SIZE))

    # Carrega o dataset
    dataset = sklearn.datasets.load_iris()
    dataset_size = len(dataset.target)

    # Inicializa as variaveis de entrada e saída
    network_input = dataset.data
    network_output = np.zeros((dataset_size, 3))
    network_output[range(dataset_size), dataset.target] = 1    # Ajusta o formato da lista com os resultados

    # Embaralha a ordem dos dados
    shuffle = np.array(range(dataset_size))
    np.random.shuffle(shuffle)    # Determina uma ordem aleatória para selecionar as entradas
    network_input = network_input[shuffle]
    network_output = network_output[shuffle]

    # Divide os dados para treinamento e para teste
    training_size = int(dataset_size * 0.8)

    training_input = network_input[:training_size]
    training_output = network_output[:training_size]

    test_input = network_input[training_size:]
    test_output = network_output[training_size:]

    # Parâmetros da rede neural
    batch_size = 1
    learning_rate = 0.01
    training_epochs = 500

    # Realiza o treinamento e obtém os pesos
    (w1, w2, w3) = run(training_input, training_output, batch_size, learning_rate, training_epochs)

    # Realiza os teste
    z1 = np.dot(test_input, w1)
    l1 = sigmoid(z1)

    z2 = np.dot(l1, w2)
    l2 = sigmoid(z2)

    z3 = np.dot(l2, w3)
    output = sigmoid(z3)

    cost = mse(test_output, output).mean()
    print(cost)
