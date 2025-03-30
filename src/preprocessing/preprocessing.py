import os
import PIL
from src.preprocessing.burgertime import apply_median_filter
from src.utils.configuration import Configuration

class Preprocessor:
    splitting_broker = {
        'saltandpepper': apply_median_filter
    }

    def __init__(self, conf: Configuration):
        self.configuration = conf

    def single_execute(self, key, image: PIL) -> PIL:
        f = Preprocessor.splitting_broker[key]
        preprocessed_image = f(image, self.configuration)
        return preprocessed_image

    def execute(self, image: PIL) -> PIL:
        retval = image
        for filter in self.configuration.get('preprocessors'):
            retval = self.single_execute(filter, retval)
        return retval