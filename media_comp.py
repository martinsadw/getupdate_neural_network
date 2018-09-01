import timeit
import random

import numpy as np


def notas_numpy(quant_participantes, quant_juizes):
  notas = np.random.randint(0, 100, (quant_participantes, quant_juizes)) / 10
  notas = np.sort(notas, axis=1)
  notas = notas[:, 1:-1]
  resultados = notas.mean(axis=1)
  melhor = resultados.argmax()

def notas_python(quant_participantes, quant_juizes):
  notas = [[random.randint(0, 100) / 10 for i in range(quant_juizes)] for i in range(quant_participantes)]
  resultados = []
  melhor = 0
  melhor_nota = 0
  for i in range(quant_participantes):
    participante = notas[i]
    participante.sort()
    participante = participante[1:-1]
    media = sum(participante) / len(participante)
    resultados.append(media)
    if media > melhor_nota:
      melhor_nota = media
      melhor = i

print(timeit.timeit("notas_numpy(100, 100)", setup="from __main__ import notas_numpy", number=1000))
print(timeit.timeit("notas_python(100, 100)", setup="from __main__ import notas_python", number=1000))
