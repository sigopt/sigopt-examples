from stanford_cars_cli import CLI, StanfordCarsCLI
from i_commandline_arguments import generate_cli_hpo


class ResNetTraining(StanfordCarsCLI):

    def __init__(self):
        super().__init__()

    def arg_parse(self):
        parser = super().arg_parse()
        parser = generate_cli_hpo(parser)
        return parser

    def get_run_arguments(self, parsed_cli_dict):
        return parsed_cli_dict, parsed_cli_dict[CLI.EPOCHS.value]


if __name__ == "__main__":
    resnet_training = ResNetTraining()
    resnet_training.run_all()
