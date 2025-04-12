''' app/utils/procesingImage.py '''

# clasificador.py
import glob
import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte


class ProcessImage():

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

        # Límites de reescalado para no saturar memoria
        self.max_width = 1000
        self.max_height = 800

    def process(self):
        
        if not os.path.isdir(self.folder_path):
            raise FileNotFoundError("No se ha encontrado el archivo")

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        image_files = []
        for ext in exts:
            image_files.extend(glob.glob(os.path.join(self.folder_path, ext)))

        if not image_files:
            raise FileExistsError("El archivo se encuentra vacio")

        all_features = []
        all_labels = []
        corrupt_files = []

        for img_path in image_files:
            image = cv2.imread(img_path)
            if image is None:
                corrupt_files.append(img_path)
                continue

            # 1) Reducir resolución
            image = self.__downscale_image(image, self.max_width, self.max_height)

            # 2) Eliminar líneas blancas / rejilla
            image = self.__remove_grid_lines(image)

            # Convertir a gris y umbralizar
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # 3) Encontrar contornos, pero ignorar subcontornos con la jerarquía
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None:
                continue

            for i, contour in enumerate(contours):
                # Ignoramos contornos que sean "hijos" (subcontornos) => subcontour
                # si `hierarchy[0][i][3] != -1` significa que tienen un contorno padre
                if hierarchy[0][i][3] != -1:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h != 0 else 0
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                # Filtrar contornos muy pequeños
                if area < 20:
                    continue
                # Filtrar contornos con relación de aspecto muy extrema
                if aspect_ratio > 4.0 or aspect_ratio < 0.25:
                    continue

                # Color medio en la región del contorno
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_color = cv2.mean(image, mask=mask)

                # Textura
                contrast, energy = self.__calculate_texture(gray)

                # Forma básica
                if 0.9 <= aspect_ratio <= 1.1:
                    shape = "Esférica/Redonda"
                elif 1.5 <= aspect_ratio <= 3.0:
                    shape = "Ovalada/Elíptica"
                else:
                    shape = "Filamentosa"

                feats = [
                    mean_color[0], mean_color[1], mean_color[2],
                    contrast, energy, area, perimeter
                ]
                all_features.append(feats)
                all_labels.append(shape)

        if not all_features or not all_labels:
            self.log_signal.emit("No se encontraron contornos válidos.")
            self.finished.emit()
            return
        
        return all_features, all_labels

    # -------------------- Funciones auxiliares --------------------

    def __downscale_image(self, img, max_w, max_h):
        """
        Reduce el tamaño de la imagen si excede el ancho/alto máximo,
        para evitar uso excesivo de memoria.
        """
        h, w = img.shape[:2]
        if w > max_w or h > max_h:
            scale_w = max_w / w
            scale_h = max_h / h
            scale = min(scale_w, scale_h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img

    def __remove_grid_lines(self, image):
        """
        Ejemplo básico de eliminación de líneas claras (rejilla) 
        mediante operaciones morfológicas.
        Ajustar si la rejilla es muy diferente.
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Umbral simple (como ejemplo, podrías tunearlo)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Kernel grande para detectar líneas horizontales
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        remove_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)

        # Kernel grande para detectar líneas verticales
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        remove_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)

        # Unir detecciones de líneas
        combined_lines = cv2.bitwise_or(remove_h, remove_v)

        # "Borramos" esas líneas del original
        # Invertimos para que las líneas blancas se vuelvan 0 y podamos hacer AND
        combined_inv = cv2.bitwise_not(combined_lines)
        # Convertimos a 3 canales para poder multiplicar con la imagen BGR
        combined_inv_bgr = cv2.cvtColor(combined_inv, cv2.COLOR_GRAY2BGR)
        cleaned = cv2.bitwise_and(image, combined_inv_bgr)

        return cleaned

    def __calculate_texture(self, gray_image):
        gray_image = img_as_ubyte(gray_image)
        glcm = graycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        return contrast, energy
