# python train_mask_detector.py --dataset dataset
# python detect_mask_video.py

# importações
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# pegando as dimenções do frame e construindo um blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# passando o blob pela rede e obtendo as detecções de rosto
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initializando a lista de rostos, a localização deles e a lista de predicoes do face mask network
	faces = []
	locs = []
	preds = []

	# fazendo um loop sobre as detecções
	for i in range(0, detections.shape[2]):
		# extraindo a probabilidade associada à detecção
		confidence = detections[0, 0, i, 2]

		# filtrando detecções fracas, garantindo que a confiança seja maior do que a confiança mínima
		if confidence > args["confidence"]:
			# calculando as coordenadas (x, y) da caixa delimitadora(bounding box) para o objeto
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# certificando que as caixas delimitadoras estejam dentro das dimensões da moldura
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extraindo o ROI do rosto convertendo-o de BGR para ordenação de canal RGB
			# e redimensionando-o para 224x224 e pré-processando-o
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# adicionando o rosto e as caixas delimitadoras às suas respectivas listas
			faces.append(face)
			locs.append((startX, startY, endX, endY))


	# só fazer previsões se pelo menos um rosto for detectado
	if len(faces) > 0:
		# para uma inferência mais rápida, faremos previsões em lote em todos os rostos ao mesmo tempo,
		# em vez de previsões uma a uma no loop `for` acima
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	#  retornando uma 2-tupla das localizações dos rostos e suas localizações correspondentes
	return (locs, preds)

# construindo o analisador de argumentos e analisando os argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# carregando o modelo de detector facial serializado do disco
print("[INFO] Carregando o modelo de detector facial...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Carregar o modelo do detector de máscara facial do disco
print("[INFO] arregando o modelo de detector facial...")
maskNet = load_model(args["model"])

# inicializandi o stream de vídeo e permitindo o sensor da câmera
print("[INFO] Iniciando o video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# fazer um loop sobre os quadros do stream de vídeo
while True:
	# pegando o quadro do stream de vídeo encadeado e redimensionando-o
	# para ter uma largura máxima de 500 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=500)

	# detectando rostos no enquadramento e analisando se eles estão usando uma máscara facial ou não
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop sobre os locais de rosto detectados e seus locais correspondentes
	for (box, pred) in zip(locs, preds):
		# descompactabdi a caixa delimitadora e as previsões
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determinando o rótulo da classe e a cor que usaremos para desenhar a caixa delimitadora e o texto
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		# incluindo a probabilidade no rótulo
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# exibindo o rótulo e o retângulo da caixa delimitadora no frame de saída
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# mostrando o frame de saída

	cv2.imshow("Projeto Bruna", frame)
	key = cv2.waitKey(1) & 0xFF

	# se a tecla "q" for pressionada, o loop será interrompido
	if key == ord("q"):
		break

# fazer um pouco de limpeza
cv2.destroyAllWindows()
vs.stop()
