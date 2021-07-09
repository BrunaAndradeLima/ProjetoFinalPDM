# importando as bibliotecas
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construindo o analisador de argumentos e analisando os argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# inicializando a taxa de aprendizagem inicial, o número de épocas para treinar e o tamanho do lote
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# Pegando a lista de imagens no diretório de conjunto de dados e inicializando a lista de dados
# (ou seja, imagens) e imagens de classe
print("[INFO] Carregando imagens...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# fazendo um loop sobre os caminhos da imagem
for imagePath in imagePaths:
	# extraindo o rótulo da classe do nome do arquivo
	label = imagePath.split(os.path.sep)[-2]

	# carregando a imagem de entrada (224x224) e o pré-processando ela
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# atualizando as listas de dados e rótulos, respectivamente
	data.append(image)
	labels.append(label)

# convertendo os dados e rótulos em matrizes NumPy
data = np.array(data, dtype="float32")
labels = np.array(labels)

# realizando codificação one-hot nas etiquetas
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# particionar os dados em divisões de treinamento e teste usando 80% dos dados para
# treinamento e os 20% restantes para teste
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# construindo o gerador de imagem de treinamento para aumento de dados
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# carregando a rede MobileNetV2, garantindo que os conjuntos de camadas head FC sejam deixados de fora
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construindo a "cabeça" do modelo que será colocado em cima do modelo base
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# colocando o modelo de cabeça FC em cima do modelo básico (este se tornará o modelo real que treinaremos)
model = Model(inputs=baseModel.input, outputs=headModel)

# fazendo um loop sobre todas as camadas no modelo básico e congelando-as para que não sejam atualizadas
# durante o primeiro processo de treinamento
for layer in baseModel.layers:
	layer.trainable = False

# compilando nosso modelo
print("[INFO] Compilando modelo...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# treinando o chefe da rede
print("[INFO] treinando head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# fazendo previsões no conjunto de teste
print("[INFO] avaliando rede...")
predIdxs = model.predict(testX, batch_size=BS)

# para cada imagem no conjunto de teste, precisamos encontrar o índice do rótulo com a maior
# probabilidade prevista correspondente
predIdxs = np.argmax(predIdxs, axis=1)

# mostrando um relatório de classificação bem formatado
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serializando o modelo para o disco
print("[INFO] salvando o modelo de detector de máscaras...")
model.save(args["model"], save_format="h5")

# traçando a perda e a precisão do treinamento
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
