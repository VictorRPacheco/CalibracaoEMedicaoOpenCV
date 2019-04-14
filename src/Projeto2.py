# -*- coding: utf-8 -*-
# Codigo para detectar e destacar cores similares
"""
-> Abre uma imagem
-> Quando clicado na imagem ideintifica onde foi o clique
-> Idenfica a cor de onde foi clicado
-> Cria matriz com True no local dos pixels a serem destacados
-> Destaca em vermelhos os pixels
Rode o codigo com -> python Projeto1.py
Para rodar os requisitos 1, 2 e 3 eh necessaria uma pasta ../data 
com arquivos de teste ou usar seus proṕrios arquivos
"""

import cv2
import numpy as np
color = None
ponto_1 = None
ponto_2 = None

# Funcao 
def Posicao_e_Distancia(event, x, y, flags, param):
	global clickPoint
	global ponto_1
	global ponto_2
	global raw
	

	if event == cv2.EVENT_LBUTTONDOWN:
		clickPoint = [x, y]
		if ponto_1 is None:
			print("set first point")
			ponto_1 = (x, y)
			raw = image.copy()
		elif ponto_2 is None:
			ponto_2 = (x, y)
		if ponto_1 is not None and ponto_2 is not None:
			cv2.line(raw, ponto_1, ponto_2, (0, 0, 255), 3)
			distancia = (((ponto_2[0] - ponto_1[0]) ** 2) + ((ponto_2[1] - ponto_1[1]) ** 2))**0.5
			print("---------------------")
			print("Comprimento da reta em pixels: ", distancia)
			ponto_1 = None
			ponto_2 = None


image = cv2.imread("../data/test.jpg")
raw = image.copy()
cv2.namedWindow("Trabalho")
cv2.setMouseCallback("Trabalho", Posicao_e_Distancia)


# Selecao de qual requisito executar
print( "╔═══════════════════════════════════════")
print( "║	Escolha um dos requisitos	")
print( "╠═══════════════════════╦═══════════════")
print( "║	Requisito 1	║ digite 1	")
print( "║	Requisito 2	║ digite 2	")
print( "║	Requisito 3	║ digite 3	")
print( "║	Requisito 4	║ digite 4	")
print( "║	Sair		║ digite 0	")
flag = input("╚═══════════════════════╩═══════════════╝\n")
"""
if flag == 1 or flag == 2:
	print( "╔═══════════════════════════════════════")
	print( "║	Escolha o arquivo de entrada	")
	print( "╠═════════════════════╦═════════════════")
	print( "║ Imagem Colorida     ║ digite 1	")
	print( "║ Escolher arquivo    ║ nome do arquivo	")
	print( "║	Sair	      ║ digite 0	")
	file = input("╚═════════════════════╩═════════════════╝\n")
	if file == '0':
		cv2.destroyAllWindows()
		flag = 0
	elif file == '1':
		image = cv2.imread("../data/test.jpg")
	else:
		image = cv2.imread(file)
	raw = image.copy()
	cv2.namedWindow("Trabalho")
	cv2.setMouseCallback("Trabalho", Posicao_e_Distancia)
"""
if flag == 1:
	print( "╔═══════════════════════════════")
	print( "║	Para sair pressione 'q'	")
	print( "╚═══════════════════════════════╝\n")
	while True:	
		cv2.imshow("Trabalho", raw)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

while True:
	cv2.imshow("Trabalho", raw)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break

cv2.destroyAllWindows()





