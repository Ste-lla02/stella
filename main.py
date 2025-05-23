import sys, os
from src.core.core_model import State
from src.preprocessing.preprocessing import Preprocessor
from src.segmentation.evaluator import MaskFeaturing
from src.segmentation.sam_segmentation import Segmenter
from src.utils.configuration import Configuration
from src.utils.utils import FileCleaner, send_ntfy_notification, send_ntfy_error,send_ntfy_start
from src.labelling.labeler import Dobby
from src.classification.mask_loader import Mask_Loader
from src.classification.image_loader import Image_Loader
from src.classification.classification import Classification
from src.classification.prediction import Prediction
from src.classification.ResNet import ResNet
import torch
import pdb


def build(conf: Configuration):
    # Cleaning
    #cleaner = FileCleaner()
    #cleaner.clean()
    # Starting
    images = State(configuration)
    topic = conf.get('ntfy_topic')
    for image_filename in images.get_base_images():
        # Preprocessing
        try:
            image_name = os.path.basename(image_filename).split('.')[0]
            image = images.get_original(image_name)
            if(not images.check_pickle(image_name)):
                preprocessor = Preprocessor(conf)
                faint_image = preprocessor.execute(image)
                images.add_preprocessed(image_name,faint_image)
                # Segmentation
                segmenter = Segmenter()
                f = MaskFeaturing()
                masks = segmenter.mask_generation(faint_image)
                masks = list(filter(lambda x: f.filter(x), masks))
                images.add_masks(image_name,masks)
                images.save_pickle(image_name)
        except Exception as e:
            send_ntfy_error(topic, image_name,str(e))
        finally:
            images.remove(image_name)
    send_ntfy_notification(topic)

def classification(conf: Configuration):
    topic = conf.get('ntfy_topic')
    try:
        send_ntfy_start(topic,'classification')
        torch.manual_seed(1)
        loader = Mask_Loader(conf,'classification')
        loader.load_data()
        resnet=ResNet(conf,loader,'classification')
        resnet()
        classifier = Classification(resnet.model, resnet.criterion, resnet.optimizer,resnet.device,conf,loader,'classification')
        best_model = classifier.train()
        classifier.evaluation_graph()
        #epoch_acc, labels_list, preds_list = classifier.test(best_model, loader.test_loader)
        #print('Test completed\n')
        #classifier.evaluate_multilabels(labels_list, preds_list)
        send_ntfy_notification(topic)
    except Exception as e:
        send_ntfy_error(topic,'classification error', str(e))
    pass


def progress(conf: Configuration):
    helper=Dobby(conf)
    helper.labeling_helper()
    pass


def clean(conf: Configuration):
    cleaner = FileCleaner()
    cleaner.clean()


def prediction(conf: Configuration):
    topic = conf.get('ntfy_topic')
    send_ntfy_start(topic, 'prediction')
    torch.manual_seed(1)
    loader=Image_Loader(conf,'prediction')
    loader.load_data()
    resnet = ResNet(conf, loader,'prediction')
    resnet()
    predictor = Prediction(resnet.model, resnet.criterion, resnet.optimizer, resnet.device, conf, loader,'prediction')
    best_model = predictor.train()
    predictor.evaluation_graph()
    send_ntfy_notification(topic)


    pass


functions = {
    'build': build,
    'clean': clean,
    'progress': progress,
    'classification':classification,
    'prediction':prediction
}

if __name__ == '__main__':
    configuration = Configuration(sys.argv[1])
    command = functions[sys.argv[2]]
    command(configuration)