import os
import torchvision
import numpy as np
from PIL import Image
import uuid
from scipy.io import loadmat
from stanford_cars_data_config import CarAugmentedDatasetAttributes
import logging
import shutil
import time
import boto3


AUGMENTED = 'augmented'
IMAGE_DATA_DIR = 'car_ims'
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def clean_up_augmented_images(path_images):
    logging.info("deleting directory: {}".format(path_images))
    shutil.rmtree(path_images)


class StanfordCarsDataAugmentation(object):

    def __init__(self, path_to_data_mat, path_images):
        self.path_images = path_images
        self.path_to_data_mat = path_to_data_mat
        self.data_mat = loadmat(self.path_to_data_mat)[CarAugmentedDatasetAttributes.ANNOTATIONS]
        self.augmented_data_mat = None
        self.rel_path_list = list()
        self.class_list = list()
        self.data_mat_rel_path = list()
        self.data_mat_labels = list()
        self.data_mat_is_augment = list()  # list of booleans indicating if the image is the original or an augment

    def augment_data(self,
                     s3_bucket_name,
                     store_to_disk,
                     store_to_s3,
                     augmentation_multiple,
                     probability,
                     brightness,
                     contrast,
                     saturation,
                     hue
                     ):

        """Augmentation multiple will be used to determine how many times to augment a single image.
        If multiple == 1, vertical flip is favored over horizontal flip.
        If multiple >=2, each image will go through horizontal and vertical flips."""

        if augmentation_multiple < 1:
            raise RuntimeError("Invalid augmentation multiple. Please specify number >=1.")

        # set up directory for augmented data
        augmented_data_directory = os.path.join(self.path_images, IMAGE_DATA_DIR, str(round(time.time())))
        if not os.path.exists(augmented_data_directory):
            logging.info("making directory for augmented data: {}".format(augmented_data_directory))
            os.makedirs(augmented_data_directory)

        s3_client = boto3.resource('s3')

        for index in range(self.data_mat.shape[1]):

            transformations = self.generate_transformation_color_jitter(augmentation_multiple=augmentation_multiple,
                                                                        probability=probability,
                                                                        brightness=brightness,
                                                                     contrast=contrast,
                                                                        saturation=saturation,
                                                                        hue=hue)

            # store data information in unnested list form
            self.unnest_data_mat(index)

            # apply transformations to image
            transformed_images_list = self.apply_transformations(index, transformations)

            if store_to_s3:
                for transformed_image in transformed_images_list:
                    # store to disk
                    transformed_image_path = self.save_transform_to_disk(index, augmented_data_directory,
                                                                       transformed_image)
                    # store to s3
                    self.save_transform_to_s3(s3_client, s3_bucket_name, transformed_image_path)
            elif store_to_disk:
                for transformed_image in transformed_images_list:
                    # store to disk
                    self.save_transform_to_disk(index, augmented_data_directory,
                                                                       transformed_image)
            else:
                logging.debug("no action specified for storing images. might be a problem for the future")

        logging.info("total number of images in directory post augmentation: {}".format(len(
            os.listdir(augmented_data_directory))))

        return augmented_data_directory

    def generate_transformation_color_jitter(self,
                                augmentation_multiple,
                                probability,
                                brightness,
                                contrast,
                                saturation,
                                hue
                                ):

        """Function will create transformations to horizontally flip and jitter color.
         A single augmentation will involve both transformations being performed."""
        transformations = list()
        for multiple in range(0, augmentation_multiple):
            horizontal_flip_transform = torchvision.transforms.RandomHorizontalFlip(probability)
            color_transform = torchvision.transforms.ColorJitter(brightness=brightness,
                                                                 contrast=contrast,
                                                                 saturation=saturation,
                                                                 hue=hue)
            transformations.append([horizontal_flip_transform, color_transform])
        return transformations

    def apply_transformations(self, index, transformations):
        """for current image, apply all transformations, save to disk, append information to data structure"""

        transformed_images_list = list()
        for transformation_group in transformations:
            composed_transform = torchvision.transforms.Compose(transformation_group)
            logging.debug("current transform: %s", composed_transform)
            image_path = os.path.join(self.path_images, self.data_mat[0][index][
                CarAugmentedDatasetAttributes.REL_IMAGE_PATH][0])
            logging.info("transforming image: {}".format(image_path))
            image = Image.open(image_path)
            transformed_image = composed_transform(image)
            transformed_images_list.append(transformed_image)

        return transformed_images_list

    def save_transform_to_disk(self, index, augmented_data_directory, transformed_image):
        augmented_image_epithet = AUGMENTED
        augmented_image_filename = '{}_{}.jpg'.format(uuid.uuid4().hex, augmented_image_epithet)
        augmented_image_filepath = os.path.join(augmented_data_directory, augmented_image_filename)

        # adding to augmented data matrix
        self.rel_path_list.append(augmented_image_filepath)
        self.class_list.append(self.data_mat[0][index][CarAugmentedDatasetAttributes.LABEL][0][0])

        # saving to disk
        transformed_image_path = os.path.join(self.path_images, augmented_image_filepath)
        logging.info("saving transformed image to: %s", transformed_image_path)

        transformed_image.save(transformed_image_path)
        return transformed_image_path

    def save_transform_to_s3(self, s3_client, s3_bucket_name, transformed_image_path):
        image_opened_s3 = open(transformed_image_path, 'rb')
        s3_client.Bucket(s3_bucket_name).put_object(Key=transformed_image_path, Body=image_opened_s3)
        image_opened_s3.close()

    def unnest_data_mat(self, index):
        """ Store matrix data as unnested lists. Converts relative path of image location to absolute path."""
        self.data_mat_rel_path.append(os.path.join(self.path_images, self.data_mat[0][index][
                                          CarAugmentedDatasetAttributes.REL_IMAGE_PATH][0]))
        self.data_mat_labels.append(self.data_mat[0][index][CarAugmentedDatasetAttributes.LABEL][0][0])
        self.data_mat_is_augment.append(False)

    def get_all_data_matrix(self):
        self.data_mat_rel_path.extend(self.rel_path_list)
        self.data_mat_labels.extend(self.class_list)
        self.data_mat_is_augment.extend([True] * len(self.rel_path_list))

        all_data_mat = np.empty(len(self.data_mat_rel_path), dtype={'names': (
            CarAugmentedDatasetAttributes.REL_IMAGE_PATH, CarAugmentedDatasetAttributes.LABEL,
            CarAugmentedDatasetAttributes.IS_AUGMENT), 'formats': ('<U200', 'uint8', '?')})
        all_data_mat[CarAugmentedDatasetAttributes.REL_IMAGE_PATH] = self.data_mat_rel_path
        all_data_mat[CarAugmentedDatasetAttributes.LABEL] = self.data_mat_labels
        all_data_mat[CarAugmentedDatasetAttributes.IS_AUGMENT] = self.data_mat_is_augment

        logging.info("total number of images for training: {}".format(len(all_data_mat)))
        return all_data_mat
