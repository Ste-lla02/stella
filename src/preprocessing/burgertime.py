import PIL
from PIL import Image, ImageFilter
from src.utils.configuration import Configuration

def apply_median_filter(pil_image: PIL, conf: Configuration):
    # https://www.crazygames.com/game/burger-time
    kernel_size = conf.get('salt_pepper_kernel')
    return pil_image.filter(ImageFilter.MedianFilter(size=kernel_size))
