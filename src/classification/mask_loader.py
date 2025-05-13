from torchvision import transforms
from src.utils.utils import cv2_to_pil
import random
from src.classification.abs_loader import AbstractLoader, MaskDataset


class Mask_Loader(AbstractLoader):
    def load_mask_dataset(self):
        self.manager.load_pickle()
        X=list()
        Y=list()
        for image_name, image in self.manager.images.items():
            masks=image['masks']
            for mask in masks:
                if('label_segmentation' in mask.keys()):
                    binary_mask=mask['segmentation']
                    mask_pillow = cv2_to_pil(binary_mask)
                    X.append(mask_pillow)
                    label_mask=mask['label_segmentation']
                    Y.append(int(label_mask)-1)
        return MaskDataset(X,Y)



    def image_generator(self,image,label,n):
        rotation=self.configuration.get('rotation_range')
        p_hor=self.configuration.get('flip_hor_probability')
        p_ver=self.configuration.get('flip_ver_probability')
        new_images=list()
        transformer=transforms.Compose([
            transforms.RandomHorizontalFlip(p=p_hor),
            transforms.RandomRotation(degrees=(0, rotation)),
            transforms.RandomVerticalFlip(p=p_ver)
        ])
        for i in range(n):
            result = transformer(image)
            new_images.append(result)
            self.dataset.add_new_instances(result,label)

        return new_images
    def augmentation(self):
        classes=list(set(self.dataset.labels))
        target_size=self.dataset.get_max_size()
        for c in classes:
            df_group, indexes = self.dataset.get_class_instances(c)
            current_size = len(df_group)

            if current_size < target_size:
                q, r = divmod(target_size, current_size)
                #combinations=[(random.choice(rotation), random.choice(width), random.choice(height)) for _ in range(repeat)]
                for index in indexes:
                    new_images = self.image_generator(self.dataset.images[index],int(c),q)

    def undersampling(self):
        classes = list(set(self.dataset.labels))
        target_size = self.dataset.get_min_size()
        for c in classes:
            df_group, indexes = self.dataset.get_class_instances(c)
            current_size = len(df_group)
            if current_size > target_size:
                q=current_size -target_size
                to_delete = random.sample(indexes, q)
                for idx in sorted(to_delete, reverse=True):
                    del self.dataset.images[idx]
                    del self.dataset.labels[idx]


