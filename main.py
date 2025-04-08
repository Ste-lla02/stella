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
    loader = Loader(conf).load_data()
    model = models.resnet50(pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    classifier = Classification(model, criterion, optimizer, data_sizes, num_epochs)
    best_model = classifier.train(train_loader)
    epoch_acc, labels_list, preds_list = classifier.test(best_model, test_loader)

    # Valutazione delle prestazioni del modello
    classifier.evaluate(labels_list, preds_list)



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