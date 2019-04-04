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
Ponto1 = None
Ponto2 = None

# Funcao 
def Posicao_e_Distancia(event, x, y, flags, param):
	global clickPoint
	global color
	global Ponto1
	global Ponto2
	

	if event == cv2.EVENT_LBUTTONDOWN:
		clickPoint = [x, y]
		color = image[y,x]
		if Ponto1 == None:
			Ponto1 = (x,y)
			print "test"
		elif Ponto2 == None:
			Ponto2 = (x,y)
		if Ponto1 is not None and Ponto2 is not None:
			cv2.line(raw, Ponto1, Ponto2, (0, 0, 255))
		print("----------------------")		
		#Identidica se a imagem eh em tons de cinza
		if color[0] == color[1] == color[2]:
			print"Posicao [X, Y]: ", (x, y), "Intensidade: ", color[0]
		else:
			print "Posicao [X, Y]: ", (x, y), "Cor: ", str(color)





# Selecao de qual requisito executar
print "╔═══════════════════════════════════════╗"
print "║	Escolha um dos requisitos	║"
print "╠═══════════════════════╦═══════════════╣"
print "║	Requisito 1	║ digite 1	║"
print "║	Requisito 2	║ digite 2	║"
print "║	Requisito 3	║ digite 3	║"
print "║	Requisito 4	║ digite 4	║"
print "║	Sair		║ digite 0	║"
flag = input("╚═══════════════════════╩═══════════════╝\n")

if flag == 1 or flag == 2:
	print "╔═══════════════════════════════════════╗"
	print "║	Escolha o arquivo de entrada	║"
	print "╠═════════════════════╦═════════════════╣"
	print "║ Imagem Colorida     ║ digite 1	║"
	print "║ Escolher arquivo    ║ nome do arquivo	║"
	print "║	Sair	      ║ digite 0	║"
	file = raw_input("╚═════════════════════╩═════════════════╝\n")
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

if flag == 1:
	print "╔═══════════════════════════════╗"
	print "║	Para sair pressione 'q'	║"
	print "╚═══════════════════════════════╝\n"
	while True:	
		cv2.imshow("Trabalho", raw)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

elif flag == 2:
	while True:
		cv2.imshow("Trabalho", raw)
		if color is not None:
			# Separa as 3 cores
			R = raw[:, :, 0].astype(np.float32)
			G = raw[:, :, 1].astype(np.float32)
			B = raw[:, :, 2].astype(np.float32)
			raw = image.copy()
			# Calcula quais pixels serão marcados com base na distância euclidiana para imagens coloridas
			if color[0] == color[1] == color[2]:
				euclidiano = ((R - color[0]) ** 2) + ((G - color[1]) ** 2) + ((B - color[2]) ** 2)
				mask = euclidiano < (13 ** 2)
			# Calcula a diferenca de intensidade para imagens em tons de cinza
			else:
				diferenca = abs(R - color[0])
				mask = diferenca < 13
			# Colore os pixels
			raw[np.where(mask)] = [0,0,255]
			color = None
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

elif flag == 3:
	print "╔═══════════════════════════════════════╗"
	print "║	Escolha o arquivo de entrada	║"
	print "╠════════════════════╦══════════════════╣"
	print "║ Vídeo Colorido     ║ digite 1		║"
	print "║ Escolher Arquivo   ║ nome do Arquivo 	║"
	print "║	Sair	     ║ digite 0		║"
	file = raw_input("╚════════════════════╩══════════════════╝\n")
	if file == '0':
		cv2.destroyAllWindows()
		flag = 0
	elif file == '1':
		cap = cv2.VideoCapture('../data/Test.avi')
	else:
		cap = cv2.VideoCapture(file)


elif flag == 4:
	cap = cv2.VideoCapture(0)

if flag == 3 or flag == 4:
	print "╔═══════════════════════════════╗"
	print "║	Para sair pressione 'q'	║"
	print "╚═══════════════════════════════╝\n"

	cv2.namedWindow("Trabalho")
	cv2.setMouseCallback("Trabalho", Posicao_e_Distancia)
	# Verifica se foi possivel abrir o arquivo
	if (cap.isOpened()== False): 
	 	print("Erro ao abrir")
	 
	# Le o video
	while(cap.isOpened()):
		ret, image = cap.read()
		if ret == True:
			raw = image.copy()
			if color is not None:
				# Separa as 3 cores
				R = raw[:, :, 0].astype(np.float32)
				G = raw[:, :, 1].astype(np.float32)
				B = raw[:, :, 2].astype(np.float32)
				raw = image.copy()
				# Calcula quais pixels serão marcados com base na distância euclidiana para imagens coloridas
				if color[0] == color[1] == color[2]:
					euclidiano = ((R - color[0]) ** 2) + ((G - color[1]) ** 2) + ((B - color[2]) ** 2)
					mask = euclidiano < (13 ** 2)
				# Calcula a diferenca de intensidade para imagens em tons de cinza
				else:
					diferenca = abs(R - color[0])
					mask = diferenca < 13
				# Colore os pixels
				raw[np.where(mask)] = [0,0,255]
			cv2.imshow('Trabalho',raw)
			if cv2.waitKey(25) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break

		else: 
			break
	cap.release()

cv2.destroyAllWindows()





