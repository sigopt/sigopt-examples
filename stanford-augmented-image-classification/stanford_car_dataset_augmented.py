import numpy as np
import torchvision
import os
from PIL import Image
import math
from a_stanford_car_dataset import AStanfordCarDataset
from stanford_cars_data_config import CarAugmentedDatasetAttributes

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class StanfordCarAugmentedDataset(AStanfordCarDataset):

    def __len__(self):
        return self.data_matrix.size

    def __getitem__(self, item):
        data_point = self.data_matrix[item]
        image_path = os.path.join(self.path_images, data_point[
            CarAugmentedDatasetAttributes.REL_IMAGE_PATH])
        # shifting labels from 1-index to 0-index
        label = data_point[CarAugmentedDatasetAttributes.LABEL] - 1
        logging.debug("image: %s is a %s with label %s", image_path, self.human_readable_labels[0, label], label)
        image = Image.open(image_path)
        if len(np.array(image).shape) < 3:
            image = image.convert("RGB")
        composed_transforms = torchvision.transforms.Compose(self.transforms)
        return {AStanfordCarDataset.TRANSFORMED_IMAGE: composed_transforms(image), AStanfordCarDataset.LABEL: label}


def preprocess_data(augmented_matdata, validation_percentage, data_subset):

    logging.info("preprocessing data")

    validation_struct = np.array([])
    training_struct = np.array([])

    unique_labels = np.unique(augmented_matdata[CarAugmentedDatasetAttributes.LABEL])

    for label in unique_labels:
        class_label = label
        class_struct = augmented_matdata[augmented_matdata[CarAugmentedDatasetAttributes.LABEL] == class_label]
        class_struct = np.sort(class_struct, order=CarAugmentedDatasetAttributes.IS_AUGMENT)

        #subset data
        class_struct = class_struct[:math.ceil(class_struct.shape[0] * data_subset)]

        # split data into training and validation
        class_struct_shape = class_struct.shape
        validation_split = math.floor(class_struct_shape[0] * validation_percentage)
        validation_data_points = class_struct[:validation_split]
        training_data_points = class_struct[validation_split:]

        # shuffle validation and training datasets for current label
        np.random.shuffle(validation_data_points)
        np.random.shuffle(training_data_points)

        if validation_struct.shape[0] == 0:
            validation_struct = validation_data_points
        else:
            validation_struct = np.append(validation_struct, validation_data_points)

        if training_struct.shape[0] == 0:
            training_struct = training_data_points
        else:
            training_struct = np.append(training_struct, training_data_points)

    # shuffle all training and validation data
    np.random.shuffle(validation_struct)
    np.random.shuffle(training_struct)

    return training_struct, validation_struct, unique_labels
