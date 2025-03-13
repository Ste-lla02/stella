import os
import numpy as np
from PIL import Image
from src.utils.configuration import Configuration
from src.preprocessing.bands_split import split_and_convert_all


"""#VARI function"""
def vari_ndvi(image, name):
    r, g, b, a = split_and_convert_all(image)
    try:
        # Calcolo del VARI
        # Evitiamo divisioni per zero aggiungendo un valore epsilon molto piccolo
        epsilon = 1e-6 #todo: estraiamolo come parametro di configurazione
        VARI = (g - r) / (g + r - b + epsilon)
        # Normalizza i valori del VARI per una migliore visualizzazione (opzionale)
        VARI_normalized = (VARI - np.min(VARI)) / (np.max(VARI) - np.min(VARI))
        # Converti in immagine e salva
        VARI_image = Image.fromarray((VARI_normalized * 255).astype(np.uint8))
        # Salva VARI_normalized
        configuration = Configuration()
        output_folder = configuration.get('splittedfolder')
        output_path = os.path.join(output_folder, f"{name}_vari_normalized.png")
        VARI_image.save(output_path)
        print(f"Immagine salvata: {output_path}")
    except Exception as e:
        print(f"VARI Calculation Error with {name}: {e}")

"""#TGI function"""
def tgi_ndvi(image, name):
    r, g, b, a = split_and_convert_all(image)
    try:
        # Calcolo del TGI
        TGI = -0.5 * (r - 0.39 * g - 0.61 * b)
        # Normalizza i valori del TGI per una migliore visualizzazione (opzionale)
        TGI_normalized = (TGI - np.min(TGI)) / (np.max(TGI) - np.min(TGI))
        # Converti in immagine e salva
        TGI_image = Image.fromarray((TGI_normalized * 255).astype(np.uint8))
        # Salva TGI_normalized
        configuration = Configuration()
        output_folder = configuration.get('splittedfolder')
        output_path = os.path.join(output_folder, f"{name}_tgi_normalized.png")
        TGI_image.save(output_path)
        print(f"Immagine salvata: {output_path}")
    except Exception as e:
        print(f"TGI Calculation Error with {name}: {e}")

"""#GLI function"""
def gli_ndvi(image, name):
    r, g, b, a = split_and_convert_all(image)
    try:
        # Calcolo del GLI
        numerator = 2 * g - r - b
        denominator = 2 * g + r + b + 1e-6  # Aggiungiamo un piccolo epsilon per evitare divisioni per zero
        GLI = numerator / denominator
        # Normalizza i valori del GLI per la visualizzazione (opzionale)
        GLI_normalized = (GLI - np.min(GLI)) / (np.max(GLI) - np.min(GLI))
        # Converti in immagine e salva
        GLI_image = Image.fromarray((GLI_normalized * 255).astype(np.uint8))
        # Salva GLI_normalized
        configuration = Configuration()
        output_folder = configuration.get('splittedfolder')
        output_path = os.path.join(output_folder, f"{name}_gli_normalized.png")
        GLI_image.save(output_path)
        print(f"Immagine salvata: {output_path}")
    except Exception as e:
        print(f"GLI Calculation Error with {name}: {e}")


"""#NGRDI function"""
def ngrdi_ndvi(image, name):
    r, g, b, a = split_and_convert_all(image)
    try:
        # Calcolo del NGRDI
        numerator = g - r
        denominator = g + r + 1e-6  # Aggiungiamo un piccolo epsilon per evitare divisioni per zero
        NGRDI = numerator / denominator
        # Normalizza i valori del NGRDI per la visualizzazione (opzionale)
        NGRDI_normalized = (NGRDI - np.min(NGRDI)) / (np.max(NGRDI) - np.min(NGRDI))
        # Converti in immagine e salva
        NGRDI_image = Image.fromarray((NGRDI_normalized * 255).astype(np.uint8))
        # Salva NGRDI_normalized
        configuration = Configuration()
        output_folder = configuration.get('splittedfolder')
        output_path = os.path.join(output_folder, f"{name}_ngrdi_normalized.png")
        NGRDI_image.save(output_path)
        print(f"Immagine salvata: {output_path}")
    except Exception as e:
        print(f"NGRDI Calculation Error with {name}: {e}")


"""#ExG function"""
def exg_ndvi(image, name):
    r, g, b, a = split_and_convert_all(image)
    try:
        # Calcolo del ExG
        ExG = 2 * g - r - b
        # Normalizza i valori del ExG per la visualizzazione (opzionale)
        ExG_normalized = (ExG - np.min(ExG)) / (np.max(ExG) - np.min(ExG))
        # Converti in immagine e salva
        ExG_image = Image.fromarray((ExG_normalized * 255).astype(np.uint8))

        # Salva NGRDI_normalized
        configuration = Configuration()
        output_folder = configuration.get('splittedfolder')
        output_path = os.path.join(output_folder, f"{name}_exg_normalized.png")
        ExG_image.save(output_path)
        print(f"Immagine salvata: {output_path}")
    except Exception as e:
        print(f"ExG Calculation Error with {name}: {e}")