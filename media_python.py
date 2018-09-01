import random


quant_participantes = 10
quant_juizes = 10

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

print("O valor das notas é: {}".format(resultados))
print("O vencedor é o participante {} com a nota {:.3f}".format(melhor + 1, melhor_nota))
