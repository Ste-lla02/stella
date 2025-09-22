import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from typing import Optional, Tuple, List
from torchvision import transforms
from pytorch_grad_cam import GradCAM as LibGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# per poter importare il modulo pytorch_grad_cam, se pycharm non lo riconosce, utilizzare il terminale: pip install -qq grad-cam==1.4.6 torchinfo==1.7.1

class GradCAM:
    """
    Incapsula la pipeline Grad-CAM:
      - preprocess coerente con il training (size, mean/std, grayscale vs RGB)
      - scelta/l'uso del layer target
      - calcolo della CAM su singola immagine
      - elaborazione di un'intera cartella (senza DataLoader)
    """
    SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        img_size: int = 224,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        grayscale: bool = True,
        device: Optional[str] = None,
    ):
        """
        Args:
            model: torch.nn.Module già inizializzato e messo in eval().
            target_layer: layer convoluzionale su cui calcolare Grad-CAM.
                          Se None, tenta una selezione automatica (ResNet ecc.).
            img_size: lato a cui ridimensionare le immagini in input al modello.
            mean/std: normalizzazione (usa i valori del training!). Se None, usa default:
                      - grayscale: mean=std=[0.5]
                      - RGB: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
            grayscale: True se il modello si aspetta 1 canale.
            device: "cuda" o "cpu". Se None, autodetect.
        """
        self.model = model
        self.device = device
        self.grayscale = grayscale
        self.img_size = img_size

        if mean is None or std is None:
            if grayscale:
                mean = [0.5]
                std = [0.5]
            else:
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
        self.mean = mean
        self.std = std

        # Preprocess coerente con training
        if self.grayscale:
            self.preprocess = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

        # Target layer
        self.target_layer = target_layer or self._auto_select_target_layer() #per noi è model.layer4[-1]


    def _auto_select_target_layer(self) -> nn.Module:
        m = self.model
        if hasattr(m, "layer4"):
            return m.layer4[-1]
        for module in reversed(list(m.modules())):
            if isinstance(module, nn.Conv2d):
                return module
        raise ValueError("Impossibile selezionare automaticamente un target_layer. Passane uno esplicitamente.")

    def _load_image_for_model(self, path: str) -> Tuple[Image.Image, np.ndarray]:
        """
        Carica l'immagine:
          - come PIL per il preprocess
          - come array RGB [H,W,3] per l'overlay (anche se grayscale)
        """
        if self.grayscale:
            pil_img = Image.open(path).convert("L")
            gray = np.array(pil_img)  # [H,W]
            rgb_overlay = np.stack([gray, gray, gray], axis=-1)  # [H,W,3]
        else:
            pil_img = Image.open(path).convert("RGB")
            rgb_overlay = np.array(pil_img)
        return pil_img, rgb_overlay

    def _to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        x = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        return x

    @torch.inference_mode()
    def predict_class(self, pil_img: Image.Image) -> int:
        """
        Predice la classe (argmax) SENZA gradienti. Utile per determinare la target_category.
        """
        logits = self.model(self._to_tensor(pil_img))
        pred_idx = int(torch.argmax(logits, dim=1).item())
        return pred_idx

    def run_on_pil(
        self,
        pil_img: Image.Image,
        rgb_overlay_np: np.ndarray,
        target_category: Optional[int] = None,
        resize_to_original: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcola Grad-CAM su una immagine già caricata.

        Returns:
            cam_out: CAM (float32 in [0,1]) ridimensionata all'originale se resize_to_original=True
            overlay: overlay RGB (uint8)
        """
        input_tensor = self._to_tensor(pil_img)

        with torch.enable_grad():
            with LibGradCAM(model=self.model, target_layers=[self.target_layer], use_cuda=(self.device == "cuda")) as cam:
                targets = None if target_category is None else [ClassifierOutputTarget(int(target_category))]
                cam_small = cam(input_tensor=input_tensor, targets=targets)[0]  # [img_size, img_size] in [0,1]

        if resize_to_original:
            H, W = rgb_overlay_np.shape[:2]
            cam_out = cv2.resize(cam_small, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            cam_out = cam_small
            rgb_overlay_np = cv2.resize(rgb_overlay_np, (cam_small.shape[1], cam_small.shape[0]), interpolation=cv2.INTER_AREA)

        rgb_float = rgb_overlay_np.astype(np.float32) / 255.0
        overlay = show_cam_on_image(rgb_float, cam_out, use_rgb=True)  # uint8 RGB
        return cam_out, overlay

    def run_on_path(
        self,
        img_path: str,
        target_category: Optional[int] = None,
        resize_to_original: bool = True,
        auto_predict_if_none: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcola Grad-CAM su un file immagine.

        Se target_category è None e auto_predict_if_none=True, predice la classe e la usa come target.
        """
        pil_img, rgb_np = self._load_image_for_model(img_path)
        if target_category is None and auto_predict_if_none:
            target_category = self.predict_class(pil_img)
        return self.run_on_pil(pil_img, rgb_np, target_category=target_category, resize_to_original=resize_to_original)

    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
        pattern_exts: Tuple[str, ...] = SUPPORTED_EXTS,
        overwrite: bool = True,
    ) -> int:
        """
        Applica Grad-CAM a tutte le immagini in input_dir e salva heatmap/overlay in output_dir.

        Salva:
          - <nome>_cam.png  (heatmap grigia 0..255)
          - <nome>_overlay.png  (overlay RGB)
        Returns:
          numero di file salvati (overlay).
        """
        os.makedirs(output_dir, exist_ok=True)
        count = 0
        for fname in sorted(os.listdir(input_dir)):
            if not fname.lower().endswith(pattern_exts):
                continue

            fpath = os.path.join(input_dir, fname)
            base, _ = os.path.splitext(fname)
            heatmap_path = os.path.join(output_dir, f"{base}_cam.png")
            overlay_path = os.path.join(output_dir, f"{base}_overlay.png")

            if (not overwrite) and os.path.exists(overlay_path) and os.path.exists(heatmap_path):
                continue

            # 1) predizione (senza grad)
            pil_img, rgb_np = self._load_image_for_model(fpath)
            with torch.inference_mode():
                pred_idx = self.predict_class(pil_img)

            # 2) CAM (con grad)
            cam, overlay = self.run_on_pil(pil_img, rgb_np, target_category=pred_idx, resize_to_original=True)

            # 3) salvataggi
            cv2.imwrite(heatmap_path, (cam * 255).astype(np.uint8))
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            count += 1

        return count
