import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm

# Rutas personalizadas según tu entorno
ROOT_DIR = "/home/cristobal/Desktop/sistema-inteligentes/datasets/Dataset_Extendido1"
SOURCE_DIRS = ["SI", "NO"]  # subcarpetas dentro de ROOT_DIR
OUT_IMAGE_DIR = os.path.join(ROOT_DIR, "data/images")
OUT_MASK_DIR = os.path.join(ROOT_DIR, "data/masks")

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

def generate_mask(image_path: str, is_alga: bool) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None or not is_alga:
        return np.zeros((224, 224), dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(blurred, 50, 150)
    return cv2.resize(edges, (224, 224))

for class_dir in SOURCE_DIRS:
    is_alga = class_dir == "SI"
    image_paths = glob(os.path.join(ROOT_DIR, class_dir, "*"))

    print(f"{class_dir}: {len(image_paths)} imágenes encontradas")

    for image_path in tqdm(image_paths, desc=f"Procesando {class_dir}"):
        filename = Path(image_path).name
        out_img = os.path.join(OUT_IMAGE_DIR, filename)
        out_mask = os.path.join(OUT_MASK_DIR, filename)

        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] No se pudo leer: {image_path}")
            continue

        resized = cv2.resize(image, (224, 224))
        mask = generate_mask(image_path, is_alga)

        cv2.imwrite(out_img, resized)
        cv2.imwrite(out_mask, mask)
