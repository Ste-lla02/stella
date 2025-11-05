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
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, filters, util
from skimage.color import rgb2gray
from skimage.restoration import denoise_bilateral

def enhance_focus_suppress_background(img, clip_limit=0.03, tile_grid_size=(8,8),
                                      highpass_radius=15, strength=1.5):
    # Assumiamo img in scala di grigi o normalizzata 0-1
    if img.ndim == 3:
        img_gray = rgb2gray(img)
    else:
        img_gray = img

    # Step 1: rimuovi rumore/sfondo leggero (optional)
    img_smooth = denoise_bilateral(img_gray, sigma_color=0.05, sigma_spatial=15)

    # Step 2: Equalizzazione locale (CLAHE)
    img_clahe = exposure.equalize_adapthist(img_smooth, clip_limit=clip_limit, nbins=256,
                                             kernel_size=tile_grid_size)

    # Step 3: Filtraggio high-pass per evidenziare dettagli
    # Metodo: originale − versione blur/low-pass
    img_low = filters.gaussian(img_clahe, sigma=highpass_radius)
    img_hp = img_clahe - img_low
    # Aumenta l’effetto
    img_hp_enh = img_clahe + strength * img_hp

    # Step 4: Normalizzazione finale
    #img_out = util.img_as_float(img_hp_enh)
    #img_out = (img_out - img_out.min()) / (img_out.max() - img_out.min())

    sigma = estimate_sigma(img_hp_enh, channel_axis=None)
    img_hp_enh = denoise_nl_means(img_hp_enh, h=1.15 * sigma, fast_mode=True)
    return img_hp_enh




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
    output_dir = Path("/Users/greeny/Desktop/Sud4VUP/input/img_SUD4VUP_complete/preprocessed_2/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # lista di tutti i file .png
    files = list(folder.glob("*.png"))

    # se vuoi i path completi in stringa
    files_name = [str(p) for p in files]
    report=open('/Users/greeny/Desktop/Sud4VUP/input/img_SUD4VUP_complete/preprocessed_2/report.txt', 'w')
    for f in files_name:
        file_name = Path(f).stem
        report.write(f"Processing {file_name}\n")
        img = load_image(f,gray=True)
        preprocessing(img,file_name,output_dir)
    report.close()


