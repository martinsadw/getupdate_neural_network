import numpy as np


quant_participantes = 10
quant_juizes = 10

notas = np.random.randint(0, 100, (quant_participantes, quant_juizes)) / 10
notas = np.sort(notas, axis=1)
notas = notas[:, 1:-1]
resultados = notas.mean(axis=1)
melhor = resultados.argmax()

print("O valor das notas é: {}".format(resultados))
print("O vencedor é o participante {} com a nota {:.3f}".format(melhor + 1, resultados[melhor]))
