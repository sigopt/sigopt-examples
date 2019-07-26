import logging
import os
from stanford_car_dataset import StanfordCarDataset, preprocess_data
from a_resnet_training_common_cli import AStanfordCarsCLI, CLI


class StanfordCarsCLI(AStanfordCarsCLI):

    def __init__(self):
        super().__init__()

    def load_datasets(self, parsed_cli_arguments):
        logging.info("loading and preprocessing data")

        training_struct, validation_struct, unique_labels = preprocess_data(
            os.path.abspath(parsed_cli_arguments[CLI.DATA.value]), validation_percentage=0.20,
            data_subset=parsed_cli_arguments[CLI.DATA_SUBSET.value])

        training_set = StanfordCarDataset(data_matrix=training_struct,
                                          path_images=os.path.abspath(parsed_cli_arguments[CLI.IMAGES.value]),
                                          transforms=AStanfordCarsCLI.IMAGE_TRANSFORMS,
                                          path_human_readable_labels=os.path.abspath(parsed_cli_arguments[
                                              CLI.LABELS.value]))

        validation_set = StanfordCarDataset(data_matrix=validation_struct,
                                            path_images=os.path.abspath(parsed_cli_arguments[CLI.IMAGES.value]),
                                            transforms=AStanfordCarsCLI.IMAGE_TRANSFORMS,
                                            path_human_readable_labels=os.path.abspath(parsed_cli_arguments[
                                                CLI.LABELS.value]))

        assert len(training_set.get_label_unique_count()[0]) == parsed_cli_arguments[CLI.NUM_CLASSES.value]

        return training_set, validation_set, os.path.abspath(parsed_cli_arguments[CLI.IMAGES.value])

    def run_all(self):
        arg_parse = self.arg_parse()
        parsed_cli = arg_parse.parse_args()
        parsed_cli_dict = parsed_cli.__dict__
        logging.debug("command line arguments: %s", parsed_cli_dict)
        training_data, validation_data, images_directory = self.load_datasets(parsed_cli_dict)
        parameter_arguments, num_epochs = self.get_run_arguments(parsed_cli_dict)
        parameter_arguments.update(parsed_cli_dict)
        self.run(parameter_arguments, num_epochs, training_data, validation_data)
