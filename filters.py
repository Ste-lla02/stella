import PIL
import matplotlib.pyplot as plt
from skimage import color, io, util, metrics, restoration
from skimage.filters import median
from skimage.morphology import disk
from skimage.restoration import denoise_bilateral
from skimage.util import img_as_float32
import numpy as np
import cv2
from pathlib import Path
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import threshold_otsu

from skimage import io, color

def load_image(path, gray=False):
    img = io.imread(path)

    if not gray:
        # restituisci in RGB “pulito”
        if img.ndim == 2:                      # già grayscale
            return color.gray2rgb(img)
        if img.ndim == 3 and img.shape[2] == 4:  # RGBA
            return color.rgba2rgb(img)
        if img.ndim == 3 and img.shape[2] == 2:  # L + alpha
            # scarta il canale alpha e “espandi” a RGB se serve
            g = img[..., 0]
            return color.gray2rgb(g)
        return img  # già RGB

    # gray == True → restituisci in scala di grigi
    if img.ndim == 2:
        return img  # già gray
    if img.ndim == 3:
        c = img.shape[2]
        if c == 3:       # RGB
            return color.rgb2gray(img)
        if c == 4:       # RGBA
            return color.rgb2gray(color.rgba2rgb(img))
        if c == 2:       # L + alpha → prendi solo la luminanza
            return img[..., 0]
    raise ValueError(f"Formato immagine non supportato: shape={img.shape}")

def conditional_inversion(image: np.ndarray) -> np.ndarray:
    # Assicura che l'immagine sia 2D
    if image.ndim == 3:
        # Se RGB, converte in grigio
        image = np.mean(image, axis=2)

    img = image.copy()

    # Se l'immagine è float [0,1], converti temporaneamente in [0,255] uint8 per Otsu
    if img.dtype != np.uint8:
        img_8bit = (img * 255).astype(np.uint8)
    else:
        img_8bit = img

    vals = img_8bit.ravel()
    if vals.size == 0:
        return img  # immagine vuota, nessuna modifica

    # Calcola soglia di Otsu
    thr = threshold_otsu(img_8bit)
    n_dark = (vals < thr).sum()
    n_light = vals.size - n_dark

    # Logica di inversione
    need_inversion = n_dark < n_light  # se prevale il chiaro, inverti
    if need_inversion:
        if img_8bit.dtype == np.uint8:
            inverted = 255 - img_8bit
        else:
            inverted = 1.0 - img  # nel caso di float [0,1]
    else:
        inverted = img_8bit

    # Ritorna nello stesso formato d’ingresso
    if image.dtype != np.uint8:
        return inverted.astype(np.float32) / 255.0
    else:
        return inverted.astype(np.uint8)

def filter_median(image, radius=2):
    """Filtro mediano per rumore impulsivo."""
    return median(image, disk(radius))


def filter_bilateral(image, sigma_color=0.05, sigma_spatial=15):
    """Filtro bilaterale che preserva i bordi."""
    return denoise_bilateral(image, sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=None)


def filter_wiener(image, balance=0.1):
    """Filtro di Wiener (restauro basato su statistica)."""
    psf = np.ones((5, 5)) / 25  # piccolo filtro medio
    return restoration.wiener(image, psf, balance=balance)



def evaluate_filter(original, denoised):
    """Calcola solo PSNR per valutare la qualità del filtraggio."""
    psnr = metrics.peak_signal_noise_ratio(original, denoised)
    return psnr

def preprocessing(img,filename,output_folder):

    # Aggiunge rumore (attivalo se vuoi testare la rimozione del rumore)
    # noisy = add_salt_pepper_noise(img, amount=0.1)
    noisy = img  # se l'immagine è già rumorosa

    # Applica filtri
    filters = {
        "Median": filter_median(noisy),
        "Bilateral": filter_bilateral(noisy),
        # "Wiener": filter_wiener(noisy)  # opzionale
    }
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # Valuta ciascun filtro
    results = {}
    iteration_filter = filters.copy()
    for name, filtered_img in iteration_filter.items():
        psnr = evaluate_filter(img, filtered_img)
        results[name] = psnr
        print(f"{name} filter -> PSNR: {psnr:.2f}")
        filtered_img = (filtered_img * 255).astype(np.uint8)
        filtered_img = clahe.apply(filtered_img)
        filtered_img = filtered_img.astype(np.float32) / 255.0
        #filtered_img=enhance_focus_suppress_background(filtered_img)
        filters[name + '+clahe'] = filtered_img

    # Mostra risultati visivi
    fig, axes = plt.subplots(1, len(filters) + 1, figsize=(15, 5))
    ax = axes.ravel()

    ax[0].imshow(noisy, cmap='gray')
    ax[0].set_title("Immagine con rumore")

    for i, (name, filtered_img) in enumerate(filters.items(), start=1):
        ax[i].imshow(filtered_img, cmap='gray')
        ax[i].set_title(name)

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    save_path = output_folder / f"{filename}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Mostra il miglior filtro in base al PSNR
    best = max(results.items(), key=lambda x: x[1])
    report.write(f"\nMiglior filtro secondo PSNR: {best[0]} (PSNR={best[1]:.2f})\n")




if __name__ == "__main__":
    # Carica immagine
    folder = Path("/Users/greeny/Desktop/Sud4VUP/input/img_SUD4VUP_complete/test/")
    output_dir = Path("/Users/greeny/Desktop/Sud4VUP/input/img_SUD4VUP_complete/preprocessed/")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(folder.glob("*.png"))  # Path objects

    report = open(output_dir / 'report.txt', 'w')
    for f in files:
        file_name = f.stem
        report.write(f"Processing {file_name}\n")

        img = load_image(str(f), gray=True)

        # True se f NON è dentro output_dir
        if not f.resolve().is_relative_to(output_dir.resolve()):
            preprocessing(img, file_name, output_dir)
    report.close()


