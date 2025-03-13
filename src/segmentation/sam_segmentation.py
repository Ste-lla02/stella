import os
import cv2, numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from src.utils.configuration import Configuration
import gc



def show_anns(masks, base_image, image_name, channel_name):
    configuration = Configuration()
    mask_folder = configuration.get("maskfolder")
    if len(masks) > 0:
        img = np.ones((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1], 4))  # crea immagine RGBA trasparente
        img[:, :, 3] = 0
        for mask in masks:
            m = mask['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])  # colore casuale per ogni maschera più trasparenza
            img[m] = color_mask
        mask_filename = os.path.join(mask_folder, f"{image_name}_{channel_name}_masks.png")
        cv2.imwrite(mask_filename, img)
        print(f"Immagine salvata: {mask_filename}")

class Segmenter:
    def __init__(self):
        configuration = Configuration()
        # Caricare il modello e assegnarlo alla CPU
        model_path = configuration.get('sam_model')
        sam_platform = configuration.get('sam_platform')
        sam = sam_model_registry["vit_h"](checkpoint=model_path)
        sam = sam.to(sam_platform)  # Cambia in "cpu" se usi CPU, "cuda" per GPU
        # Getting mask quality parameter values
        points_per_side = configuration.get('points_per_side')
        min_mask_quality = configuration.get('min_mask_quality')
        min_mask_stability = configuration.get('min_mask_stability')
        layers = configuration.get('layers')
        crop_n_points_downscale_factor = configuration.get('crop_n_points_downscale_factor')
        min_mask_region_area = configuration.get('min_mask_region_area')
        self.max_mask_region_area = configuration.get('max_mask_region_area')
        # Creare il generatore di maschere
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=min_mask_quality,
            stability_score_thresh=min_mask_stability,
            crop_n_layers=layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area
        )

    def mask_generation(self, image, name, channel):
        configuration = Configuration()
        mask_folder = configuration.get("maskfolder")
        splitted_folder = configuration.get("splittedfolder")
        scaling_factor = configuration.get("image_scaling")
        gc.collect()
        file_to_open = os.path.join(splitted_folder, f"{name}_{channel}.png")
        image = cv2.imread(file_to_open)  # Sostituisci con il percorso corretto
        colored_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        width = int(image.shape[1] * scaling_factor)
        height = int(image.shape[0] * scaling_factor)
        colored_image = cv2.resize(colored_image, (width, height), interpolation=cv2.INTER_AREA)
        masks = self.mask_generator.generate(colored_image)
        # Salvare le maschere
        for i, mask in enumerate(masks):
            mask_image = mask["segmentation"].astype(np.uint8) * 255  # Convertire in immagine binaria
            mask_filename = os.path.join(mask_folder, f"{name}_{channel}_mask_{i}.png")
            cv2.imwrite(mask_filename, mask_image)
            print(f"Immagine salvata: {mask_filename}")
        # Save the overall figure
        show_anns(masks, image, name, channel)
        pass

    @staticmethod
    def mask_voting(mask_list):
        pass


'''#mask_generator = SamAutomaticMaskGenerator( model=sam, points_per_side=8, pred_iou_thresh=0.7, stability_score_thresh=0.8)
      # Riduci il numero di punti (valore predefinito è 32)
      # Imposta una soglia inferiore per la qualità delle maschere
      # Riduci la soglia di stabilità

def process_image(image_path, output_drive_path, scale_percent=20):
    """
    Processa un'immagine: la ridimensiona, genera le maschere, le salva singolarmente e crea
    un'immagine con tutte le maschere sovrapposte.

    Parametri:
    - image_path: percorso completo dell'immagine originale
    - output_drive_path: cartella in cui salvare i risultati su Google Drive
    - scale_percent: percentuale di ridimensionamento dell'immagine
    """

    # Leggere l'immagine
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Estrarre il nome del file senza estensione
    image_name = os.path.basename(image_path).split('.')[0]  # Es. "201506_cropped"

    # Ridimensionare l'immagine
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Generare le maschere
    masks.pop(0)

    # Creare la cartella per le maschere
    mask_folder = os.path.join(output_drive_path, f"{image_name}_masks")
    os.makedirs(mask_folder, exist_ok=True)

    # Salvare le maschere
    for i, mask in enumerate(masks):
        mask_image = mask["segmentation"].astype(np.uint8) * 255  # Convertire in immagine binaria
        mask_filename = os.path.join(mask_folder, f"{image_name}_mask_{i}.png")
        cv2.imwrite(mask_filename, mask_image)

    print(f"Salvate {len(masks)} maschere in {mask_folder}")

    # Creare l'immagine con maschere sovrapposte
    display_and_save_masks(image_resized, masks, output_drive_path, image_name)

def display_and_save_masks(image, masks, output_drive_path, image_name):
    """
    Sovrappone tutte le maschere all'immagine originale e la salva.
    """

    # Creare una mappa di colori distinti per le maschere
    colors = list(mcolors.TABLEAU_COLORS.values())
    combined_mask = np.zeros_like(image, dtype=np.float32)

    # Sovrapporre le maschere con colori unici
    for i, mask in enumerate(masks):
        color = np.array(mcolors.to_rgb(colors[i % len(colors)]))
        segmentation = mask["segmentation"].astype(np.float32)
        for c in range(3):
            combined_mask[:, :, c] += segmentation * color[c]

    # Normalizzare la maschera combinata
    combined_mask = np.clip(combined_mask, 0, 1)

    # Creare la figura
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Immagine originale
    axes[0].imshow(image)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    # Immagine con maschere sovrapposte
    axes[1].imshow(image, alpha=0.7)
    axes[1].imshow(combined_mask, alpha=0.2)
    axes[1].set_title("Masked Image")
    axes[1].axis("off")

    plt.tight_layout()

    # Salvare l'immagine con le maschere sovrapposte
    masked_image_path = os.path.join(output_drive_path, f"{image_name}_masked.png")
    plt.savefig(masked_image_path, format='png', dpi=300)
    plt.close()

    print(f"Immagine con maschere salvata in {masked_image_path}")

# Specificare il percorso della cartella con le immagini su Google Drive
input_folder = "/content/drive/MyDrive/Colab Notebooks/Mat4Pat/GEP/Kenya/cropped"

# Specificare il percorso della cartella in cui salvare i risultati
output_folder = "/content/drive/MyDrive/Colab Notebooks/Mat4Pat/GEP/Kenya/masks"

# Creare la cartella di output se non esiste
os.makedirs(output_folder, exist_ok=True)

# Ottenere la lista delle immagini nella cartella
image_files = [f for f in os.listdir(input_folder) if f.endswith((".png"))]

# Elaborare tutte le immagini
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    process_image(image_path, output_folder)

"""#SAM with points

Le coordinate (x, y) sono espresse in pixel rispetto all'immagine
"""

from segment_anything import sam_model_registry, SamPredictor

# Carica il modello SAM con un checkpoint pre-addestrato
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")

# Crea un oggetto per fare previsioni
predictor = SamPredictor(sam)

import cv2
import numpy as np

# Carica l'immagine
image = cv2.imread("/content/201812_cropped.png")

# Imposta l'immagine per il modello
predictor.set_image(image)

input_points = np.array([[2000, 435]])
input_labels = np.array([1])  # 1 significa "inclusione" dell'area

masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True  # Se impostato, restituisce più maschere di segmentazione
)

masks.shape

import matplotlib.pyplot as plt

# Visualizza l'immagine originale
plt.figure(figsize=(10, 5))
plt.imshow(image)  # L'immagine deve essere in formato RGB
plt.axis("off")

# Disegna le maschere sopra l'immagine
for mask in masks:
    plt.imshow(mask, alpha=0.4, cmap="Greens")  # Overlay della maschera
plt.title("Maschere segmentate")
plt.show()

import cv2
import numpy as np

# Load image and mask
image = cv2.imread("image.jpg")  # Replace with your image path
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale

# Find contours of the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # Get the bounding square envelope
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop the square envelope from the original image
    cropped_square = image[y:y+h, x:x+w]

    # Create a blank mask for applying the irregular crop
    mask_inv = np.zeros_like(cropped_square, dtype=np.uint8)
    # Draw the filled contour on the cropped mask
    cv2.drawContours(mask_inv, contours, -1, (255, 255, 255), thickness=cv2.FILLED, offset=(-x, -y))

    # Convert to a binary mask
    mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_BGR2GRAY)

    # Apply the irregular mask to crop the desired region
    cropped_image = cv2.bitwise_and(cropped_square, cropped_square, mask=mask_inv)

    # Save or show results
    cv2.imwrite("cropped_square.jpg", cropped_square)
    cv2.imwrite("cropped_irregular.jpg", cropped_image)

    cv2.imshow("Cropped Square", cropped_square)
    cv2.imshow("Cropped Irregular", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''