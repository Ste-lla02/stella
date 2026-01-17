import PIL
import matplotlib.pyplot as plt
from skimage import color, io, util, metrics, restoration
from skimage.filters import median
from skimage.morphology import disk
from skimage.restoration import denoise_bilateral
import numpy as np
import cv2
from pathlib import Path
import matplotlib.patches as patches
from skimage import io, color
import PIL
from src.preprocessing.burgertime import apply_median_filter
from src.utils.configuration import Configuration

class Preprocessor:

    def __init__(self, conf: Configuration):
        self.configuration = conf
        self.output_dir_selection = conf.get('preprocessedselection')
        self.output_dir_selection.mkdir(parents=True, exist_ok=True)
        self.output_dir_preprocessed = conf.get('preprocessedfolder')
        self.output_dir_preprocessed.mkdir(parents=True, exist_ok=True)

    def execute(self, image:PIL,image_name:str) -> PIL:

        image_name= image_name+'.png'
        preprocessed_image = self.preprocessing(image,image_name)
        return preprocessed_image



    def rms_contrast(self,img):
        """Calcola l'RMS contrast (deviazione standard normalizzata)."""
        img = img.astype(np.float32)
        return np.std(img) / np.mean(img)


    def image_quality_index(self,original, enanched):
        mu_x = np.mean(original)
        mu_y = np.mean(enanched)
        sigma_x = np.var(original)
        sigma_y = np.var(enanched)
        sigma_xy = np.mean((original - mu_x) * (enanched - mu_y))

        numerator = 4 * sigma_xy * mu_x * mu_y
        denominator = (mu_x**2 + mu_y**2) * (sigma_x + sigma_y)

        return numerator / denominator




    #buono alpha=0.15,beta=0.65,gamma=0.20
    def combined_quality_metric(self,original, denoised, alpha=0.15,beta=0.65,gamma=0.20):
        # Calcolo PSNR (in dB)
        psnr_val = self.psnr(original, denoised)

        # Normalizziamo il PSNR su scala [0,1] assumendo 0–100 dB come range utile
        psnr_norm = np.clip(psnr_val / 100.0, 0, 1)


        # Calcolo contrasto normalizzato
        contrast_val = self.rms_contrast(denoised)
        #contrast_norm = np.clip(contrast_val / 0.3, 0, 1)  # 0.5 ≈ contrasto alto tipico

        #calcolo IQI
        IQI = self.image_quality_index(original, denoised)
        #IQI_norm = (IQI + 1) / 2.0

        # Media pesata
        Q = alpha * psnr_norm + beta * contrast_val + gamma * IQI
        return Q
    def load_image(self,path, gray=False):
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



    def filter_median(self,image, radius=2):
        """Filtro mediano per rumore impulsivo."""
        return median(image, disk(radius))


    def filter_bilateral(self,image, sigma_color=0.05, sigma_spatial=15):
        """Filtro bilaterale che preserva i bordi."""
        return denoise_bilateral(image, sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=None)


    def filter_wiener(self,image, balance=0.1):
        """Filtro di Wiener (restauro basato su statistica)."""
        psf = np.ones((5, 5)) / 25  # piccolo filtro medio
        return restoration.wiener(image, psf, balance=balance)



    def psnr(self,original, denoised):
        """Calcola solo PSNR per valutare la qualità del filtraggio."""
        psnr = metrics.peak_signal_noise_ratio(original, denoised)
        return psnr

    def preprocessing(self,img,filename):

        # Aggiunge rumore (attivalo se vuoi testare la rimozione del rumore)
        # noisy = add_salt_pepper_noise(img, amount=0.1)
        noisy = img  # se l'immagine è già rumorosa

        # Applica filtri
        filters = {
            "Median": self.filter_median(noisy),
            "Bilateral": self.filter_bilateral(noisy),
            # "Wiener": filter_wiener(noisy)  # opzionale
        }
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Valuta ciascun filtro
        results = {}
        iteration_filter = filters.copy()
        print('Image '+str(filename)+'\n')
        for name, filtered_img in iteration_filter.items():
            #FILTRO
            metric = self.combined_quality_metric(img, filtered_img)
            results[name] = metric
            print(f"{name} filter -> metric: {metric:.2f}")
            #CLAHE
            filtered_clahe = (filtered_img * 255).astype(np.uint8)
            filtered_clahe = clahe.apply(filtered_clahe)
            filtered_clahe = filtered_clahe.astype(np.float32) / 255.0
            filters[name + '+clahe'] = filtered_clahe
            metric = self.combined_quality_metric(img, filtered_clahe)
            results[name+'+clahe'] = metric
            print(f"{name+'+clahe'} filter -> metric: {metric:.2f}")
            #HE
            filtered_he = (filtered_img * 255).astype(np.uint8)
            filtered_he = cv2.equalizeHist(filtered_he)
            filtered_he = filtered_he.astype(np.float32) / 255.0
            filters[name + '+he'] = filtered_he
            metric = self.combined_quality_metric(img, filtered_he)
            results[name + '+he'] = metric
            print(f"{name + '+he'} filter -> metric: {metric:.2f}")
        best = max(results.items(), key=lambda x: x[1])
        #report.write(f"\nMiglior filtro secondo metric: {best[0]} (metric={best[1]:.2f})\n")
        best_image_name=best[0]
        flag=False
        if(self.psnr(img,filters[best[0]])<0.55):
            filters[best[0]]=self.filter_bilateral(filters[best[0]])
            flag=True

        # Mostra risultati visivi
        fig, axes = plt.subplots(1, len(filters) + 1, figsize=(15, 5))
        ax = axes.ravel()

        ax[0].imshow(noisy, cmap='gray')
        ax[0].set_title("Immagine con rumore")

        for i, (name, filtered_img) in enumerate(filters.items(), start=1):

            ax[i].imshow(filtered_img, cmap='gray')
            if name==best_image_name:
                if(flag):
                    name=name+' improved'
                h, w = filtered_img.shape[:2]

                # Crea un rettangolo verde attorno all'immagine
                rect = patches.Rectangle(
                    (0, 0),  # coordinate angolo in basso a sinistra
                    w, h,  # larghezza e altezza
                    linewidth=2,  # spessore bordo
                    edgecolor='lime',  # colore bordo (verde acceso)
                    facecolor='none'  # nessun riempimento
                )

                # Aggiungi il rettangolo all’asse
                ax[i].add_patch(rect)
            ax[i].set_title(name)

        for a in ax:
            a.axis('off')

        plt.tight_layout()
        save_path = self.output_dir_selection / f"{filename}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        best_img = filters[best[0]]
        #save_path_best = self.output_dir / f"{filename}.png"
        #plt.imsave(save_path_best, best_img, cmap='gray')
        plt.close(fig)
        return best_img















