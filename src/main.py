import sys, os
from PIL import Image
from src.preprocessing.image_cropper import crop_image_with_polygon
from src.preprocessing.preprocessing import splitting_broker
from src.segmentation.sam_segmentation import Segmenter
from src.utils.configuration import Configuration

if __name__ == '__main__':
    configuration = Configuration(sys.argv[1])
    input_dir = configuration.get('imagefolder')
    image_type = configuration.get('imagetype')
    filenames = list(os.listdir(input_dir))
    filenames = list(filter(lambda x: x.lower().endswith((image_type)), filenames))
    images = {}
    for filename in filenames:
        # Preliminaries
        image_path = os.path.join(input_dir, filename)
        image_name = os.path.basename(image_path).split('.')[0]
        images[image_name] = {}
        image = Image.open(image_path)
        images[image_name]['original'] = image
        # Cropping
        cropped_image = crop_image_with_polygon(image, image_name)
        images[image_name]['cropped'] = cropped_image
        # Splitting
        channels = configuration.get('channels')
        for channel in channels:
            function = splitting_broker[channel]
            splitted = function(cropped_image, image_name)
            images[image_name][channel] = splitted
        # Segmentation
        masks = list()
        segmenter = Segmenter()
        for channel in channels:
            base = images[image_name][channel]
            mask = segmenter.mask_generation(base, image_name, channel)
            masks.append(mask)
        final_mask = Segmenter.mask_voting(masks)









