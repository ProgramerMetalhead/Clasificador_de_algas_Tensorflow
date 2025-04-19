# unet_data_generator.py
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import sequence

class UNetDataGenerator(Sequence):
    def __init__(self, image_paths, mask_dir, image_size=(224, 224), batch_size=4, shuffle=True):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y = self.__load_batch(batch_paths)
        return X, Y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_paths)

    def __load_batch(self, batch_paths):
        X = []
        Y = []
        for img_path in batch_paths:
            mask_path = os.path.join(self.mask_dir, os.path.basename(img_path))
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            img = cv2.resize(img, self.image_size).astype("float32") / 255.0
            mask = cv2.resize(mask, self.image_size)
            mask = (mask > 0).astype("float32")[..., np.newaxis]
            X.append(img)
            Y.append(mask)
        return np.array(X), np.array(Y)


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from glob import glob
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (224, 224)
IMAGE_DIR = "/home/cristobal/Desktop/sistema-inteligentes/datasets/Dataset_Extendido1/data/images"
MASK_DIR = "/home/cristobal/Desktop/sistema-inteligentes/datasets/Dataset_Extendido1/data/masks"
BATCH_SIZE = 8
EPOCHS = 10
WEIGHTS_PATH = "/home/cristobal/Desktop/sistema-inteligentes/datasets/Dataset_Extendido1/unet_weights.h5"

def load_image_mask_pairs(image_paths):
    X, Y = [], []
    for img_path in image_paths:
        mask_path = os.path.join(MASK_DIR, os.path.basename(img_path))
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        img = cv2.resize(img, IMAGE_SIZE).astype("float32") / 255.0
        mask = cv2.resize(mask, IMAGE_SIZE)
        mask = (mask > 0).astype("float32")  # binaria

        X.append(img)
        Y.append(mask[..., np.newaxis])
    return np.array(X), np.array(Y)

def build_unet(input_shape=(224, 224, 3)):
    def dbl_conv(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return x

    inputs = layers.Input(shape=input_shape)
    c1 = dbl_conv(inputs, 16); p1 = layers.MaxPool2D()(c1)
    c2 = dbl_conv(p1, 32); p2 = layers.MaxPool2D()(c2)
    c3 = dbl_conv(p2, 64)
    u4 = layers.UpSampling2D()(c3); u4 = layers.Concatenate()([u4, c2]); c4 = dbl_conv(u4, 32)
    u5 = layers.UpSampling2D()(c4); u5 = layers.Concatenate()([u5, c1]); c5 = dbl_conv(u5, 16)
    output = layers.Conv2D(1, 1, activation="sigmoid")(c5)
    return models.Model(inputs, output)

# Cargar datos
image_paths = glob(os.path.join(IMAGE_DIR, "*"))
train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
X_train, Y_train = load_image_mask_pairs(train_paths)
X_val, Y_val = load_image_mask_pairs(val_paths)

# Compilar modelo
model = build_unet(input_shape=(224, 224, 3))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Entrenar
model.fit(X_train, Y_train,
          validation_data=(X_val, Y_val),
          batch_size=BATCH_SIZE,
          epochs=EPOCHS)

# Guardar
model.save_weights(WEIGHTS_PATH)
print(f"✅ Pesos guardados en {WEIGHTS_PATH}")
