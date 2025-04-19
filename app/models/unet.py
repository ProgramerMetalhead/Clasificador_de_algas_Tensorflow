# app/models/unet.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_unet(input_shape=(224, 224, 3), num_classes=1):
    """
    Construye una U-Net simple para segmentación binaria.
    :param input_shape: Dimensiones de entrada (H,W,Canales)
    :param num_classes: 1 para binario, más si hay segmentación multi-clase
    :return: Modelo Keras compilado (sin entrenar)
    """
    inputs = keras.Input(shape=input_shape)

    # --- [Bloque de downsampling] ---
    # Bloque 1
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPool2D((2, 2))(c1)

    # Bloque 2
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPool2D((2, 2))(c2)

    # Bloque 3 (bottleneck)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)

    # --- [Bloque de upsampling] ---
    # Up Bloque 1
    u4 = layers.UpSampling2D((2, 2))(c3)
    u4 = layers.Concatenate()([u4, c2])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(c4)

    # Up Bloque 2
    u5 = layers.UpSampling2D((2, 2))(c4)
    u5 = layers.Concatenate()([u5, c1])
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(c5)

    # Capa final => salida binaria [0..1]
    if num_classes == 1:
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)
    else:
        # Salida multi-clase con softmax
        outputs = layers.Conv2D(num_classes, 1, activation='softmax')(c5)

    model = keras.Model(inputs, outputs)
    return model
