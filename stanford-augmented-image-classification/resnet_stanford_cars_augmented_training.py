from stanford_cars_augmentation_cli import StanfordCarsAugmentationCLI, CLI
from i_commandline_arguments import generate_cli_hpo_augment, generate_cli_hpo


class ResNetAugmentedTraining(StanfordCarsAugmentationCLI):

    def __init__(self):
        super().__init__()

    def arg_parse(self):
        parser = super().arg_parse()
        parser = generate_cli_hpo(parser)
        parser = generate_cli_hpo_augment(parser)
        return parser

    def get_run_arguments(self, parsed_cli_dict):
        return parsed_cli_dict, parsed_cli_dict[CLI.EPOCHS.value]


if __name__ == "__main__":
    resnet_augmented_training = ResNetAugmentedTraining()
    resnet_augmented_training.run_all()
