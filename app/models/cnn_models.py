# app/models/cnn_models.py
"""Arquitecturas CNN – solo clasificación (sin carga de datos).

Define 4 modelos ligeros y un helper `build_model` que los compila.
El pre‑procesamiento de imágenes se realiza externamente (ProcessImage).
"""

from __future__ import annotations

from typing import Tuple, Dict
import tensorflow as tf
from tensorflow.keras import layers, models

# ------------------------------------------------------------------
# Arquitecturas
# ------------------------------------------------------------------

def SimpleCNN(input_shape: Tuple[int, int, int], num_classes: int, feat_size: int = 64):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(feat_size, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="SimpleCNN")


def VGGMini(input_shape: Tuple[int, int, int], num_classes: int, feat_size: int = 128):
    def vgg_block(x, f):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        return layers.MaxPool2D()(x)

    inputs = layers.Input(shape=input_shape)
    x = vgg_block(inputs, 32)
    x = vgg_block(x, 64)
    x = layers.Flatten()(x)
    x = layers.Dense(feat_size, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="VGGMini")


def ResNetMini(input_shape: Tuple[int, int, int], num_classes: int, _: int = 0):
    def res_block(x, f):
        shortcut = x
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(f, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        if shortcut.shape[-1] != f:
            shortcut = layers.Conv2D(f, 1, padding="same")(shortcut)
        x = layers.Add()([x, shortcut])
        return layers.Activation("relu")(x)

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = res_block(x, 32)
    x = res_block(x, 64)
    x = res_block(x, 64)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="ResNetMini")


def MobileNetMini(input_shape: Tuple[int, int, int], num_classes: int, _: int = 0):
    def dw_sep(x, f, s=1):
        x = layers.DepthwiseConv2D(3, strides=s, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        x = layers.Conv2D(f, 1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU(6.0)(x)

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, strides=2, padding="same")(inputs)
    x = layers.ReLU(6.0)(x)

    x = dw_sep(x, 32)
    x = dw_sep(x, 64, s=2)
    x = dw_sep(x, 64)
    x = dw_sep(x, 128, s=2)
    x = dw_sep(x, 128)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="MobileNetMini")


# Mapa de constructores
CNN_FACTORY: Dict[str, callable] = {
    "CNN‑Simple": SimpleCNN,
    "CNN‑VGGMini": VGGMini,
    "CNN‑ResMini": ResNetMini,
    "CNN‑MobileMini": MobileNetMini,
}


# Helper -----------------------------------------------------------

def build_model(model_id: str, input_size: int, num_classes: int, feat_size: int) -> tf.keras.Model:
    """Devuelve un modelo compilado según parámetros del widget.

    Parameters
    ----------
    model_id : str
        Clave en `CNN_FACTORY`.
    input_size : int
        Ancho/alto de entrada (cuadrado).
    num_classes : int
        Número de clases.
    feat_size : int
        Tamaño de la capa densa interna (solo usado por Simple/VGG).
    """
    if model_id not in CNN_FACTORY:
        raise ValueError(f"Modelo CNN no soportado: {model_id}")
    model = CNN_FACTORY[model_id]((input_size, input_size, 3), num_classes, feat_size)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
