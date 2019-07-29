import logging
import shutil
import orchestrate.io

from a_resnet_training_common_cli import AStanfordCarsCLI, CLI
from enum import Enum
from stanford_augmented_data_processor import StanfordAugmentedDataProcessor
import os
import torchvision


class AugmentHyperparameters(Enum):
    BRIGHTNESS = 'brightness'
    CONTRAST = 'contrast'
    SATURATION = 'saturation'
    HUE = 'hue'


class AugmentCLI(Enum):
    AUGMENT_MULTIPLIER = 'multiplier'
    STORE_TO_DISK = 'store_to_disk'
    STORE_TO_S3 = 'store_to_s3'
    PROBABILITY = 'probability'
    S3_BUCKET_NAME = 's3_bucket_name'


class StanfordCarsAugmentationCLI(AStanfordCarsCLI):

    def __init__(self):
        super().__init__()

    def arg_parse(self):
        """Adding cli option for augmented hyperparameter tuning."""
        parser = super().arg_parse()
        parser.add_argument("--" + AugmentCLI.AUGMENT_MULTIPLIER.value, dest=AugmentCLI.AUGMENT_MULTIPLIER.value, type=int,
                            help="augmentation multiplier to use. recommended max = 5", required=True)

        parser.add_argument("--" + AugmentCLI.PROBABILITY.value, dest=AugmentCLI.PROBABILITY.value,
                            type=float,
                            help="probability of flipping image to use. recommended = 1.0", required=True)

        parser.add_argument("--" + AugmentCLI.S3_BUCKET_NAME.value, dest=AugmentCLI.S3_BUCKET_NAME.value, type=str,
                            help="name of S3 bucket to store augmented images. please make sure you have read/write "
                                 "access to this bucket.")

        parser.add_argument("--" + AugmentCLI.STORE_TO_DISK.value, dest=AugmentCLI.STORE_TO_DISK.value, action='store_true',
                            help="whether or not to store to disk")

        parser.add_argument("--" + "no-" + AugmentCLI.STORE_TO_DISK.value, dest=AugmentCLI.STORE_TO_DISK.value,
                            action='store_false',
                            help="whether or not to store to disk")

        parser.add_argument("--" + AugmentCLI.STORE_TO_S3.value, dest=AugmentCLI.STORE_TO_S3.value, action='store_true',
                            help="whether or not to store to S3")

        parser.add_argument("--" + "no-" + AugmentCLI.STORE_TO_S3.value, dest=AugmentCLI.STORE_TO_S3.value,
                            action='store_false',
                            help="whether or not to store to S3")
        return parser

    def get_hyperparameter_tuple(self, augment_hyperparameters_enum, dataset_parameters):
        hyperparameter_value = dataset_parameters[augment_hyperparameters_enum.value]
        return hyperparameter_value, hyperparameter_value

    def load_datasets(self, parsed_cli_arguments, dataset_parameters):

        logging.info("loading and preprocessing data with augmentation")
        stanford_augmented_data_processor = StanfordAugmentedDataProcessor(
            path_images=os.path.abspath(parsed_cli_arguments[CLI.IMAGES.value]),
            transforms=[torchvision.transforms.Resize(224),
                                                      torchvision.transforms.RandomCrop(224),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                       [0.229, 0.224, 0.225])],

            path_human_readable_labels=os.path.abspath(parsed_cli_arguments[
                                                           CLI.LABELS.value]))
        augmented_data_matrix, augmented_data_directory = stanford_augmented_data_processor.augment_data(
            path_to_data_mat=os.path.abspath(parsed_cli_arguments[CLI.DATA.value]),
            augmentation_multiple=parsed_cli_arguments[
                AugmentCLI.AUGMENT_MULTIPLIER.value],
            s3_bucket_name=parsed_cli_arguments[AugmentCLI.S3_BUCKET_NAME.value],
            store_to_s3=parsed_cli_arguments[AugmentCLI.STORE_TO_S3.value],
            store_to_disk=parsed_cli_arguments[AugmentCLI.STORE_TO_DISK.value],
            probability=parsed_cli_arguments[AugmentCLI.PROBABILITY.value],
            brightness=self.get_hyperparameter_tuple(
                AugmentHyperparameters.BRIGHTNESS, dataset_parameters),
            contrast=self.get_hyperparameter_tuple(
                AugmentHyperparameters.CONTRAST, dataset_parameters),
            saturation=self.get_hyperparameter_tuple(
                AugmentHyperparameters.SATURATION, dataset_parameters),
            hue=self.get_hyperparameter_tuple(
                AugmentHyperparameters.HUE, dataset_parameters))

        training_struct, validation_struct, unique_labels = stanford_augmented_data_processor.preprocess_data(
            augmented_data_matrix,
            0.20, parsed_cli_arguments[CLI.DATA_SUBSET.value])
        training_set = stanford_augmented_data_processor.get_data_generator(data_matrix=training_struct)
        validation_set = stanford_augmented_data_processor.get_data_generator(data_matrix=validation_struct)

        return training_set, validation_set, augmented_data_directory

    def run_all(self):
        arg_parse = self.arg_parse()
        parsed_cli = arg_parse.parse_args()
        parsed_cli_dict = parsed_cli.__dict__
        logging.debug("command line arguments: %s", parsed_cli_dict)
        parameter_assignments, num_epochs = self.get_run_arguments(parsed_cli_dict)
        training_data, validation_data, augmented_data_directory = self.load_datasets(
            parsed_cli_arguments=parsed_cli_dict, dataset_parameters=parameter_assignments)
        orchestrate.io.log_metadata('augmented_directory_name', augmented_data_directory)
        logging.info("augmentation data directory at: {}".format(augmented_data_directory))
        parameter_assignments.update(parsed_cli_dict)
        self.run(parameter_assignments, num_epochs, training_data, validation_data)
        if parsed_cli_dict[AugmentCLI.STORE_TO_DISK.value] is False or parsed_cli_dict[AugmentCLI.STORE_TO_S3.value] \
            is True:
            logging.info("deleting augmented data directory stored on disk: {}".format(augmented_data_directory))
            shutil.rmtree(augmented_data_directory)
