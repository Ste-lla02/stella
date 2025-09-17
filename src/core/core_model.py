import os
from PIL import Image
import cv2, numpy as np
import pickle
from src.utils.utils import cv2_to_pil, pil_to_cv2
import glob
error_list=open('errors.txt','w')
class State:
    def __init__(self, conf):
        self.input_directory = conf.get('srcfolder')
        self.filetype = conf.get('imagetype')
        self.preprocessed_directory = conf.get('preprocessedfolder')
        self.mask_directory = conf.get("maskfolder")
        self.pickle = conf.get('picklefolder')
        self.save_flag = conf.get('save_images')
        self.clean()

    def clean(self):
        filenames = list(os.listdir(self.input_directory))
        filenames = list(filter(lambda x: x.lower().endswith((self.filetype)), filenames))
        self.images = dict()
        for filename in filenames:
            image_name = os.path.basename(filename).split('.')[0]
            self.images[image_name] = {}
            image_path = os.path.join(self.input_directory,filename)
            image = Image.open(image_path)
            self.images[image_name]['original'] = image
            self.images[image_name]['masks'] = list()
            self.images[image_name]['preprocessed'] = None


    def make_overall_image(self, image_name, masks,highlits=False):
        blended = None
        try:
            base_image = self.get_original(image_name)
        except KeyError as exception:
            base_image=Image.open(self.input_directory + os.sep + image_name+self.filetype)

        finally:
            if len(masks) > 0:
                h, w = masks[0]['segmentation'].shape
                overlay = np.zeros((h, w, 3), dtype=np.uint8)  # Immagine vuota per le maschere
                for mask in masks:
                    mask_img = mask['segmentation'].astype(np.uint8)  # Converti la maschera in uint8 (0-1 -> 0-255)
                    if len(masks)==1 or (highlits==True):
                        color=[255, 0, 0] #rosso se ho solo una mask
                    else:
                        color = np.random.randint(0, 255, (3,), dtype=np.uint8)  # Colore casuale
                    overlay[mask_img > 0] = color  # Applica colore alla maschera
                if base_image is not None:
                    base_img = pil_to_cv2(base_image)
                    alpha = 0.35
                    blended = cv2.addWeighted(base_img, 1 - alpha, overlay, alpha, 0)

                else:
                    blended = overlay
            else: blended = base_image


        return blended

    def make_masks_overlay(self, masks):
        h, w = masks[0]['segmentation'].shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)  # Immagine nera di base

        for mask in masks:
            mask_img = mask['segmentation'].astype(np.uint8)
            color = [255, 255, 255]  # BIANCO
            overlay[mask_img > 0] = color

        return overlay

    def remove(self, image_name: str):
        del self.images[image_name]

    def get_base_images(self):
        return list(self.images.keys())

    def get_original(self, image_name: str) -> Image:
        return self.images[image_name]['original']

    def set_original(self, image_name: str, image: Image):
        self.images[image_name]['original'] = image

    def get_channel(self, image_name: str, channel_name: str) -> Image:
        return self.images[image_name][channel_name]

    def add_preprocessed(self, image_name, image):
        self.images[image_name]['preprocessed'] = image
        filename = f"{image_name}_preprocessed.png"
        self.save_image_and_log(image,self.preprocessed_directory,filename)

    def add_mask(self, image_name, mask):
        self.images[image_name]['masks'].append(mask)
        filename = f"{image_name}_mask_{mask['id']}.png"
        mask_pillow = cv2_to_pil(mask['segmentation'])
        self.save_image_and_log(mask_pillow, self.mask_directory, filename)

    def add_masks(self, image_name, masks):
        for mask in masks:
            self.add_mask(image_name, mask)
        merged = self.make_overall_image(image_name, masks)
        self.images[image_name]['merged'] = merged
        merged_pillow = cv2_to_pil(merged)
        filename = f"{image_name}_mergedmasks.png"
        self.save_image_and_log(merged_pillow, self.mask_directory, filename)

    def save_image_and_log(self, image, directory, filename):
        if self.save_flag:
            output_path = os.path.join(directory, filename)
            image.save(output_path)
            print(f"Immagine salvata: {output_path}")

    def save_pickle(self, image_name):
        filename = f'{image_name}.pickle'
        output_path = os.path.join(self.pickle,filename)
        with open(output_path, "wb") as f:
            pickle.dump(self.images[image_name], f)

    def check_pickle(self, image_name):
        filename = f'{image_name}.pickle'
        pattern = os.path.join(self.pickle, filename)
        check = glob.glob(pattern)
        return check

    def load_pickle(self):
        self.images = dict()
        filenames = list(os.listdir(self.pickle))
        for filename in filenames:
            image_name = os.path.splitext(filename)[0]
            input_path = os.path.join(self.pickle, filename)
            try:
                with open(input_path, "rb") as f:
                    temp = pickle.load(f)
                    self.images[image_name] = temp
            except Exception as e:
                print('ERROR pickle ' + image_name + '\n')
                print(str(e) + '\n')

