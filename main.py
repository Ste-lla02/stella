import sys, os
from src.core.core_model import State
from src.preprocessing.preprocessing import Preprocessor
from src.segmentation.evaluator import MaskFeaturing
from src.segmentation.sam_segmentation import Segmenter
from src.utils.configuration import Configuration
from src.utils.utils import FileCleaner, send_ntfy_notification, send_ntfy_error,send_ntfy_start
from src.labelling.labeler import Dobby
from src.classification.loader import Loader
import torch.optim as optim
import torch.nn as nn
from torchvision import models
from src.classification.Classification import Classification
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
            send_ntfy_error(topic, image_name,str(e))
        finally:
            images.remove(image_name)
    send_ntfy_notification(topic)

def classification(conf: Configuration):
    topic = conf.get('ntfy_topic')
    try:
        send_ntfy_start(topic,'classificiation')
        print('Start..\n')
        torch.manual_seed(1)
        loader = Loader(conf)
        loader.load_data()
        print('Dataset loaded\n')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(512, loader.dataset.get_num_classes())
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=conf.get('learning_rate'))
        classifier = Classification(model, criterion, optimizer, device,conf,loader)
        print('Model instanciated\n')
        best_model = classifier.train()
        print('Trial completed\n')
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