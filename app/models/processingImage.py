# app/utils/processingImage.py
"""ProcessImage – U‑Net‑powered pre‑processor
================================================
Compatibilidad total con la GUI:
  • Si se instancian `log_signal` y `finished` desde un QThread, los usa; de lo
    contrario escribe en stdout.
  • Lee parámetros editables por el usuario desde `AppConfig` (on‑missing‑weights,
    extra‑features, target_size).
  • Soporta CPU/GPU de forma automática, pero sólo usará GPU si el modelo es
    CNN y el dispositivo está disponible (colocamos placeholder para esa lógica
    por si se integra con otros CNNs en el futuro).

Puntos implementados:
  3️⃣  Gestión estricta de pesos – aborta o usa random según config.
  4️⃣  Extra‑features opcionales: mask_area, aspect_ratio, mean_intensity.
  5️⃣  Tamaño objetivo de entrada configurable (mantiene relación de aspecto).
  6️⃣  Docstrings + type hints → estilo NumPy.
  7️⃣  Cacheo de la U‑Net a nivel de clase para acelerar ejecuciones múltiples.
"""

from __future__ import annotations

import glob
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from app.utils.config import AppConfig


class ProcessImage:
    """Segmenta microalgas con una U‑Net y genera un *dataset* JSON.

    Parameters
    ----------
    folder_path : str
        Carpeta fuente con subcarpetas de imágenes.
    output_json : str, default "dataset.json"
        Fichero donde se guardará la lista de registros.
    processed_dir : str, default "processed_images"
        Carpeta donde se escriben las máscaras binarizadas.
    model_weights : str, default "unet_weights.h5"
        Pesos pre‑entrenados. El comportamiento ante ausencia está regulado por
        `AppConfig.ON_MISSING_WEIGHTS`.
    log_signal : pyqtSignal | None, default None
        Señal opcional para emitir logs en la GUI.
    finished_signal : pyqtSignal | None, default None
        Señal opcional para notificar fin de proceso en la GUI.
    """

    # ------------------------------------------------------------------
    #  Clase‑nivel: cache de modelos para no reconstruirlos repetidamente
    # ------------------------------------------------------------------
    _model_cache: Dict[Tuple[int, int, int], tf.keras.Model] = {}

    def __init__(
        self,
        folder_path: str,
        output_json: str = "dataset.json",
        processed_dir: str = "processed_images",
        model_weights: str = "unet_weights.h5",
        log_signal=None,
        finished_signal=None,
    ) -> None:
        # Rutas & config
        self.folder_path = os.path.abspath(folder_path)
        self.output_json = output_json
        self.processed_dir = processed_dir
        self.model_weights = model_weights

        self.on_missing = getattr(AppConfig, "ON_MISSING_WEIGHTS", "error")
        self.extra_feats = bool(getattr(AppConfig, "EXTRA_FEATURES", False))
        self.target_size: int = int(getattr(AppConfig, "TARGET_SIZE", 224))

        # Señales opcionales (para conectar con QThread)
        self._log_signal = log_signal
        self._finished_signal = finished_signal

        os.makedirs(self.processed_dir, exist_ok=True)

        # Modelo (cacheado)
        input_shape = (self.target_size, self.target_size, 3)
        if input_shape not in self._model_cache:
            self._model_cache[input_shape] = self._build_unet(input_shape)
        self.model = self._model_cache[input_shape]

        # Cargar pesos
        if os.path.isfile(self.model_weights):
            self.model.load_weights(self.model_weights)
            self._log(f"[INFO] Pesos cargados: {self.model_weights}")
        else:
            if self.on_missing == "error":
                raise FileNotFoundError(
                    f"Pesos no encontrados y ON_MISSING_WEIGHTS='error': {self.model_weights}")
            self._log(
                f"[WARN] No se encontró '{self.model_weights}'. La U‑Net usará pesos aleatorios.")

    # ------------------------------------------------------------------
    #  Helper de logging compatible GUI / consola
    # ------------------------------------------------------------------
    def _log(self, msg: str):
        if self._log_signal is not None:
            self._log_signal.emit(msg)
        else:
            print(msg)

    # ------------------------------------------------------------------
    #  U‑Net mini + cache
    # ------------------------------------------------------------------
    @staticmethod
    def _dbl_conv(x, filters: int):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return x

    def _build_unet(self, input_shape: Tuple[int, int, int]):
        inputs = layers.Input(shape=input_shape)
        c1 = self._dbl_conv(inputs, 16); p1 = layers.MaxPool2D()(c1)
        c2 = self._dbl_conv(p1, 32);   p2 = layers.MaxPool2D()(c2)
        c3 = self._dbl_conv(p2, 64)                                # bottleneck
        u4 = layers.UpSampling2D()(c3); u4 = layers.Concatenate()([u4, c2])
        c4 = self._dbl_conv(u4, 32)
        u5 = layers.UpSampling2D()(c4); u5 = layers.Concatenate()([u5, c1])
        c5 = self._dbl_conv(u5, 16)
        outputs = layers.Conv2D(1, 1, activation="sigmoid")(c5)
        return models.Model(inputs, outputs, name="mini_unet")

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def process(self) -> List[Dict]:
        """Procesa la carpeta y genera el JSON con registros.

        Returns
        -------
        list[dict]
            Lista de registros con nombre, ruta, label, feats y processed_path.
        """
        if not os.path.isdir(self.folder_path):
            raise FileNotFoundError(f"Carpeta no encontrada: {self.folder_path}")

        # Recolectar archivos
        img_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            img_paths += glob.glob(os.path.join(self.folder_path, "**", ext), recursive=True)
        if not img_paths:
            raise FileExistsError("No se encontraron imágenes soportadas.")

        records: List[Dict] = []
        for path in img_paths:
            img = cv2.imread(path)
            if img is None:
                self._log(f"[WARN] No se pudo leer: {path}")
                continue

            mask = self._predict_mask(img)
            proc_name = f"{uuid.uuid4().hex}_{Path(path).name}"
            proc_path = os.path.join(self.processed_dir, proc_name)
            cv2.imwrite(proc_path, mask)

            feats: Dict[str, float] = {"mask_coverage": float(np.mean(mask > 0))}
            if self.extra_feats:
                feats.update(self._extra_features(img, mask))

            label = Path(path).parent.name  # sub‑carpeta como label
            records.append({
                "name": Path(path).name,
                "path": path,
                "label": label,
                "feats": feats,
                "processed_path": proc_path,
            })

        # Guardar JSON
        with open(self.output_json, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=4, ensure_ascii=False)
        self._log(f"[INFO] Dataset guardado en {self.output_json} – {len(records)} registros")

        if self._finished_signal is not None:
            self._finished_signal.emit()
        return records

    # ------------------------------------------------------------------
    #  Inferencia + feats helpers
    # ------------------------------------------------------------------
    def _resize_keep_ratio(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) == self.target_size:
            return cv2.resize(img, (self.target_size, self.target_size))
        scale = self.target_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.zeros((self.target_size, self.target_size, 3), dtype=img.dtype)
        y0 = (self.target_size - nh) // 2
        x0 = (self.target_size - nw) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        return canvas

    def _predict_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        img_bgr = self._resize_keep_ratio(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inp = img_rgb.astype("float32") / 255.0
        inp = np.expand_dims(inp, 0)
        pred = self.model.predict(inp, verbose=0)[0, ..., 0]
        return (pred > 0.5).astype(np.uint8) * 255

    def _extra_features(self, img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        area = int(np.count_nonzero(mask))
        yx = np.column_stack(np.where(mask > 0))
        if yx.size:
            y0, x0 = yx.min(axis=0)
            y1, x1 = yx.max(axis=0)
            aspect = (x1 - x0 + 1) / (y1 - y0 + 1)
            mean_int = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[mask > 0]))
        else:
            aspect = 0.0
            mean_int = 0.0
        return {"mask_area": area, "bbox_aspect": aspect, "mean_intensity": mean_int}
