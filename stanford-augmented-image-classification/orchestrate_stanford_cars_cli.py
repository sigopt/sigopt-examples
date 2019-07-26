import math
from stanford_cars_cli import CLI, StanfordCarsCLI
from a_resnet_training_common_cli import Hyperparameters
from i_orchestrate_multitask import get_assignments


class OrchestrateCLI(StanfordCarsCLI):
    def __init__(self):
        super().__init__()

    def get_run_arguments(self, parsed_cli_arguments):
        orchestrate_assignments, percentage_epochs = get_assignments(Hyperparameters)
        num_epochs = math.ceil(parsed_cli_arguments[CLI.EPOCHS.value] * percentage_epochs)
        return orchestrate_assignments, num_epochs


if __name__ == "__main__":
    orchestrate_cli = OrchestrateCLI()
    orchestrate_cli.run_all()
