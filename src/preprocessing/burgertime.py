import PIL
from PIL import Image, ImageFilter
from src.utils.configuration import Configuration
import cv2
import numpy as np

def apply_median_filter(pil_image: PIL, conf: Configuration):
    # https://www.crazygames.com/game/burger-time

    if pil_image.mode != 'L':
        gray = pil_image.convert('L')
    else:
        gray = pil_image

    # Converti in array NumPy
    img_np = np.array(gray)

    # Crea CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Applica CLAHE (input deve essere numpy.ndarray in uint8)
    cl1 = clahe.apply(img_np)

    # Converti di nuovo in immagine PIL
    img_clahe_pil = Image.fromarray(cl1)

    # Prendi kernel dal conf (default 3 se non presente)
    kernel_size = conf.get('salt_pepper_kernel')

    # Applica filtro mediano PIL
    result = img_clahe_pil.filter(ImageFilter.MedianFilter(size=kernel_size))
    return result
