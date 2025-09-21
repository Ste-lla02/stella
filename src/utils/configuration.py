import configparser
from configparser import ConfigParser
from src.utils import metaclasses
import ast
from src.utils.metaclasses import Singleton


class Configuration(metaclass=Singleton):
    def __init__(self, inifilename):
        self.board = dict()
        self.load(inifilename)

    def get(self, key):
        return self.board[key]

    def reset(self, inifilename):
        self.board = dict()
        self.load(inifilename)

    def put(self, key, value):
        self.board[key] = value

    def clean(self):
        self.board = dict()

    @staticmethod
    def tolist(value):
        retval = value
        if len(retval) > 0:
            retval = value.split(',')
            retval = list(map(lambda x: x.strip(),retval))
        return retval

    def loadSection(self, reader, s):
        temp = dict()
        options = []
        try:
            options = reader.options(s)
        except configparser.NoSectionError as nse:
            temp[s] = ['*']
        for o in options:
            try:
                value = reader[s][o]
                temp[s] = self.tolist(value)
            except:
                print("exception on %s!" % o)
        return temp

    def loadOption(self, reader, option):
        temp = dict()
        options = reader.options(option)
        for o in options:
            value = reader[option][o]
            temp[o] = self.tolist(value)
        retval = dict()
        retval[option] = temp
        return retval

    def load(self, inifile):
        reader = ConfigParser()
        reader.read(inifile)
        try:
            # Main
            temp = reader['main'].get('srcfolder',None)
            self.put('srcfolder', temp)
            temp = reader['main'].get('preprocessedfolder',1)
            self.put('preprocessedfolder', temp)
            temp = reader['main'].get('maskfolder',1)
            self.put('maskfolder', temp)
            temp = reader['main'].get('image_scaling',1)
            self.put('image_scaling', float(temp))
            temp = reader['main'].get('picklefolder', None)
            self.put('picklefolder', temp)
            temp = reader['main'].get('lablescsv', None)
            self.put('lablescsv', temp)
            temp = reader['main'].get('codescsv', None)
            self.put('codescsv', temp)
            temp = reader['main'].get('reportcsv', None)
            self.put('reportcsv', temp)
            temp = reader['main'].get('ntfy_topic', None)
            self.put('ntfy_topic', temp)
            temp = reader['main'].get('imagetype', None)
            self.put('imagetype', temp)
            # SAM
            temp = reader['sam'].get('sam_model', None)
            temp = self.tolist(temp)
            self.put('sam_model', temp[0])
            self.put('sam_kind', temp[1])
            temp = reader['sam'].get('sam_platform', 'cpu')
            self.put('sam_platform', temp)
            temp = reader['sam'].get('points_per_side', 32)
            self.put('points_per_side', int(temp))
            temp = reader['sam'].get('min_mask_quality', 0.8)
            self.put('min_mask_quality', float(temp))
            temp = reader['sam'].get('min_mask_stability', 0.9)
            self.put('min_mask_stability', float(temp))
            temp = reader['sam'].get('layers', 1)
            self.put('layers', int(temp))
            temp = reader['sam'].get('crop_n_points_downscale_factor', 2)
            self.put('crop_n_points_downscale_factor', int(temp))
            temp = reader['sam'].get('min_mask_region_area', 50)
            self.put('min_mask_region_area', int(temp))
            # Preprocessing
            temp = reader['preprocessing'].get('preprocessors', [])
            temp = self.tolist(temp)
            self.put('preprocessors', temp)
            temp = reader['preprocessing'].get('salt_pepper_kernel', 3)
            self.put('salt_pepper_kernel', int(temp))
            # Filtering
            temp = reader['filters'].get('filters', [])
            temp = self.tolist(temp)
            self.put('filters', temp)
            temp = reader['filters'].get('min_roundness', 0)
            self.put('min_roundness', float(temp))
            temp = reader['filters'].get('max_roundness', 1)
            self.put('max_roundness', float(temp))
            temp = reader['filters'].get('min_eccentricity', 0)
            self.put('min_eccentricity', float(temp))
            temp = reader['filters'].get('max_eccentricity', 1)
            self.put('max_eccentricity', float(temp))
            temp = reader['filters'].get('min_iou', 0)
            self.put('min_iou', float(temp))
            temp = reader['filters'].get('max_iou', 1)
            self.put('max_iou', float(temp))
            temp = reader['filters'].get('min_stability', 0)
            self.put('min_stability', float(temp))
            temp = reader['filters'].get('max_stability', 1)
            self.put('max_stability', float(temp))
            temp = reader['filters'].get('min_meters', 0)
            self.put('min_meters', float(temp))
            temp = reader['filters'].get('max_meters', 10000)
            self.put('max_meters', float(temp))
            temp = reader['filters'].get('max_percentage', 0)
            self.put('max_percentage', float(temp))
            temp = reader['filters'].get('min_percentage', 100)
            self.put('min_percentage', float(temp))
            temp = reader['filters'].get('max_pixels', 0)
            self.put('max_pixels', float(temp))
            temp = reader['filters'].get('min_pixels', 100000)
            self.put('min_pixels', float(temp))
            #control
            temp = reader['control'].get('save_images', True)
            temp = temp == 'True'
            self.put('save_images', temp)
            temp = reader['control'].get('clean_images', True)
            temp = temp == 'True'
            self.put('clean_images', temp)
            temp = reader['control'].get('clean_pickles', True)
            temp = temp == 'True'
            self.put('clean_pickles', temp)
            #Labelling
            temp = reader['labelling'].get('df_pickle', None)
            self.put('df_pickle', temp)
            temp = reader['labelling'].get('df_masks', None)
            self.put('df_masks', temp)

            # Augmentation
            temp = reader['augmentation'].get('rotation_range', 0)
            self.put('rotation_range', int(temp))
            temp = reader['augmentation'].get('flip_hor_probability', 0)
            self.put('flip_hor_probability', float(temp))
            temp = reader['augmentation'].get('flip_ver_probability', 0)
            self.put('flip_ver_probability', float(temp))
            # Classification
            temp = reader['classification'].get('classification_split_option', None)
            self.put('split_option', temp)
            temp = reader['classification'].get('test_split', 0.2)
            self.put('classification_test_split', float(temp))
            temp = reader['classification'].get('num_epochs', 100)
            self.put('classification_num_epochs', int(temp))
            temp = reader['classification'].get('learning_rate', 1e-4)
            self.put('classification_learning_rate', float(temp))
            temp = reader['classification'].get('batch_size', 32)
            self.put('classification_batch_size', int(temp))
            temp = reader['classification'].get('preprocessing','')
            self.put('classification_preprocessing', str(temp))
            temp = reader['classification'].get('delta', 0.0)
            self.put('classification_delta', float(temp))
            temp = reader['classification'].get('patience', 5)
            self.put('classification_patience', int(temp))
            temp = reader['classification'].get('verbose', True)
            temp = temp == 'True'
            self.put('classification_verbose', temp)
            temp = reader['classification'].get('model_path', 'best_model.pth')
            self.put('classification_model_path', str(temp))
            # Prediction
            temp = reader['prediction'].get('split_option', None)
            self.put('prediction_split_option', temp)
            temp = reader['prediction'].get('split_file', None)
            self.put('split_file', temp)
            temp = reader['prediction'].get('test_split', 0.2)
            self.put('prediction_test_split', float(temp))
            temp = reader['prediction'].get('num_epochs', 100)
            self.put('prediction_num_epochs', int(temp))
            temp = reader['prediction'].get('learning_rate', 1e-4)
            self.put('prediction_learning_rate', float(temp))
            temp = reader['prediction'].get('batch_size', 32)
            self.put('prediction_batch_size', int(temp))
            temp = reader['prediction'].get('preprocessing', '')
            self.put('prediction_preprocessing', str(temp))
            temp = reader['prediction'].get('delta', 0.0)
            self.put('prediction_delta', float(temp))
            temp = reader['prediction'].get('patience', 5)
            self.put('prediction_patience', int(temp))
            temp = reader['prediction'].get('verbose', True)
            temp = temp == 'True'
            self.put('prediction_verbose', temp)
            temp = reader['prediction'].get('model_path', 'best_model.pth')
            self.put('prediction_model_path', str(temp))

        except Exception as s:
            print(s)
