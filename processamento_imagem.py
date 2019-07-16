# coding=utf-8
import numpy as np
import cv2
import mahotas


def escreve(img, texto, cor=(255, 0, 0)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, texto, (10, 20), fonte, 0.5, cor, 0,
                cv2.LINE_AA)


imgColorida = cv2.imread('dados.png')

# Passo 1: Conversão para tons de cinza
img = cv2.cvtColor(imgColorida, cv2.COLOR_BGR2GRAY)

# Passo 2: Blur/Suavização da imagem
suave = cv2.blur(img, (7, 7))

# Passo 3: Binarização resultando em pixels brancos e pretos
T = mahotas.thresholding.otsu(suave)
bin = suave.copy()
bin[bin > T] = 255
bin[bin < 255] = 0
bin = cv2.bitwise_not(bin)

# Passo 4: Detecção de bordas com Canny
bordas = cv2.Canny(bin, 70, 150)

# Passo 5: Identificação e contagem dos contornos da imagem
# cv2.RETR_EXTERNAL = conta apenas os contornos externos
(objetos,lx) = cv2.findContours(bordas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# A variável lx (lixo) recebe dados que não são utilizados

escreve(img, "Imagem em tons de cinza", 0)
escreve(suave, "Suavizacao com Blur", 0)
escreve(bin, "Binarizacao com Metodo Otsu", 255)
escreve(bordas, "Detector de bordas Canny", 255)

temp = np.vstack([
    np.hstack([img, suave]),
    np.hstack([bin, bordas])
])

cv2.imshow("Quantidade de objetos: " + str(len(objetos[0])), temp)
imgC2 = imgColorida.copy()
cv2.imshow("Imagem Original", imgColorida)

cv2.drawContours(imgC2, objetos, -1, (255, 0, 0), 2)
escreve(imgC2, str(len(objetos)) + " objetos encontrados!")
cv2.imshow("Resultado", imgC2)
cv2.waitKey(0)