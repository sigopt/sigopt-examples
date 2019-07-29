from scipy.io import loadmat
import numpy as np
import torchvision
import os
from PIL import Image
import math
from a_stanford_car_dataset import AStanfordCarDataset

import logging

from stanford_cars_data_config import CarDatasetAttributes

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class StanfordCarDataset(AStanfordCarDataset):

    def __len__(self):
        return self.data_matrix.size

    def __getitem__(self, item):
        data_point = self.data_matrix[item]
        image_path = os.path.join(self.path_images, data_point[CarDatasetAttributes.REL_IMAGE_PATH][0][0])
        # shifting labels from 1-index to 0-index
        label = data_point[CarDatasetAttributes.LABEL][0][0][0] - 1
        logging.debug("image: %s is a %s with label %s", image_path, self.human_readable_labels[0, label], label)
        image = Image.open(image_path)
        if len(np.array(image).shape) < 3:
            image = image.convert("RGB")
        composed_transforms = torchvision.transforms.Compose(self.transforms)
        return {AStanfordCarDataset.TRANSFORMED_IMAGE: composed_transforms(image), AStanfordCarDataset.LABEL: label}


def preprocess_data(path_to_matdata, validation_percentage, data_subset):

    logging.info("preprocessing data")
    data_struct = loadmat(path_to_matdata)
    annotations = data_struct[CarDatasetAttributes.ANNOTATIONS]
    annotations_labels = annotations[CarDatasetAttributes.LABEL]

    validation_struct = np.array([])
    training_struct = np.array([])

    unique_labels = np.unique(annotations_labels)

    for label in unique_labels:
        class_label = label[0][0]
        class_struct = annotations[annotations[CarDatasetAttributes.LABEL] == class_label]
        class_struct = np.reshape(class_struct, (class_struct.shape[0], 1))
        np.random.shuffle(class_struct)

        #subset data
        class_struct = class_struct[:math.ceil(class_struct.shape[0] * data_subset)]

        # split data into training and validation
        class_struct_shape = class_struct.shape
        validation_split = math.floor(class_struct_shape[0] * validation_percentage)
        validation_data_points = class_struct[:validation_split]
        training_data_points = class_struct[validation_split:]

        if validation_struct.shape[0] == 0:
            validation_struct = validation_data_points
        else:
            validation_struct = np.append(validation_struct, validation_data_points)

        if training_struct.shape[0] == 0:
            training_struct = training_data_points
        else:
            training_struct = np.append(training_struct, training_data_points)

    # shuffle training and validation data
    validation_struct = np.reshape(validation_struct, (validation_struct.shape[0],1))
    np.random.shuffle(validation_struct)

    training_struct = np.reshape(training_struct, (training_struct.shape[0],1))
    np.random.shuffle(training_struct)

    return training_struct, validation_struct, unique_labels
