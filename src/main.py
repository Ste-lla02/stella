import sys, os
from src.core.core_model import State
from src.preprocessing.preprocessing import Preprocessor
from src.segmentation.evaluator import MaskFeaturing
from src.segmentation.sam_segmentation import Segmenter
from src.utils.configuration import Configuration
from src.utils.metautils import FileCleaner

def build(conf: Configuration):
    # Cleaning
    cleaner = FileCleaner()
    cleaner.clean()
    # Starting
    images = State(configuration)
    for image_filename in images.get_base_images():
        # Preprocessing
        image_name = os.path.basename(image_filename).split('.')[0]
        image = images.get_original(image_name)
        preprocessor = Preprocessor(conf)
        faint_image = preprocessor.execute(image)
        images.add_preprocessed(image_name,faint_image)
        # Segmentation
        segmenter = Segmenter()
        f = MaskFeaturing()
        masks = segmenter.mask_generation(faint_image)
        masks = list(filter(lambda x: f.filter(x), masks))
        images.add_masks(image_name,masks)
    images.save_pickle()
    pass

def progress(conf: Configuration):
    images = State(conf)
    images.load_pickle()
    pass

functions = {
    'build': build,
    'progress': progress
}

if __name__ == '__main__':
    configuration = Configuration(sys.argv[1])
    command = functions[sys.argv[2]]
    command(configuration)