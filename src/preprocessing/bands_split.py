import os

import numpy as np
from PIL import Image
from src.utils.configuration import Configuration

def split_and_convert_all(image):
    r, g, b, a = image.split()
    r = np.array(r, dtype=np.float32)
    g = np.array(g, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    a = np.array(a, dtype=np.float32)
    return r, g, b, a

#todo: fare funzione per il salvaaggio immagine unica!
def red(image, image_name):
    r, _, _, _ = image.split()
    configuration = Configuration()
    output_folder = configuration.get('splittedfolder')
    output_path = os.path.join(output_folder, f"{image_name}_red.png")
    r.save(output_path)
    print(f"Immagine salvata: {output_path}")
    return r

def green(image, image_name):
    _, g, _, _ = image.split()
    configuration = Configuration()
    output_folder = configuration.get('splittedfolder')
    output_path = os.path.join(output_folder, f"{image_name}_green.png")
    g.save(output_path)
    print(f"Immagine salvata: {output_path}")
    return g

def blue(image, image_name):
    _, _, b, _ = image.split()
    configuration = Configuration()
    output_folder = configuration.get('splittedfolder')
    output_path = os.path.join(output_folder, f"{image_name}_blue.png")
    b.save(output_path)
    print(f"Immagine salvata: {output_path}")
    return b

def alpha(image, image_name):
    _, _, _, a = image.split()
    configuration = Configuration()
    output_folder = configuration.get('splittedfolder')
    output_path = os.path.join(output_folder, f"{image_name}_alpha.png")
    a.save(output_path)
    print(f"Immagine salvata: {output_path}")
    return a
