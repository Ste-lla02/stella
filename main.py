import sys, os
from src.core.core_model import State
from src.preprocessing.preprocessing import Preprocessor
from src.segmentation.evaluator import MaskFeaturing
from src.segmentation.sam_segmentation import Segmenter
from src.utils.configuration import Configuration
from src.utils.utils import FileCleaner, send_ntfy_notification, send_ntfy_error
from src.labelling.labeler import Dobby
from src.classification.loader import Loader
import torch.optim as optim
import torch.nn as nn
from torchvision import models
from src.classification import Classification
import torch


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
            send_ntfy_error(topic, image_name)
            send_ntfy_error(topic,str(e))
        finally:
            images.remove(image_name)
    send_ntfy_notification(topic)

def classification(conf: Configuration):
    torch.manual_seed(1)
    loader = Loader(conf)
    loader.load_data()
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(512, loader.dataset.get_num_classes())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=conf.get('learning_rate'))
    classifier = Classification(model, criterion, optimizer, loader.dataset_sizes, conf.get('num_epochs'))
    best_model = classifier.train(loader.train_loader)
    epoch_acc, labels_list, preds_list = classifier.test(best_model, loader.test_loader)
    classifier.evaluate_multilabels(labels_list, preds_list)
    pass



    #images.load_pickle()
    pass
def progress(conf: Configuration):
    images = State(conf)
    helper=Dobby(conf)
    helper.labeling_helper()

    #images.load_pickle()
    pass


def clean(conf: Configuration):
    cleaner = FileCleaner()
    cleaner.clean()


functions = {
    'build': build,
    'clean': clean,
    'progress': progress,
    'classification':classification
}

if __name__ == '__main__':
    configuration = Configuration(sys.argv[1])
    command = functions[sys.argv[2]]
    command(configuration)