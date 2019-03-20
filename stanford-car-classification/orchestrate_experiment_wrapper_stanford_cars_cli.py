from resnet import get_pretrained_resnet
import logging
from enum import Enum
from torch.utils.data import DataLoader
import torch
from resnet import PalmNet
import orchestrate.io
import numpy as np
import math
from resnet_stanford_cars_cli import StanfordCarsCLI, Hyperparameters, CLI


class OrchestrateCLI(StanfordCarsCLI):
    def __init__(self):
        super().__init__()

    def run(self, parsed_cli_arguments, training_data, validation_data):
        class Multitask(Enum):
            full = parsed_cli_arguments[CLI.EPOCHS.value]
            medium = math.ceil(parsed_cli_arguments[CLI.EPOCHS.value] * 0.50)
            small = math.ceil(parsed_cli_arguments[CLI.EPOCHS.value] * 0.10)

        logging.info("loading pretrained model and establishing model characteristics")

        logging.debug("sigopt assignments being used: %s",
                      {hyperparameter.value: orchestrate.io.assignment(hyperparameter.value) for hyperparameter in
                       Hyperparameters})

        resnet_pretrained_model = get_pretrained_resnet(parsed_cli_arguments[CLI.FREEZE_WEIGHTS.value],
                                                        parsed_cli_arguments[CLI.NUM_CLASSES.value],
                                                        parsed_cli_arguments[CLI.MODEL.value])
        cross_entropy_loss = torch.nn.CrossEntropyLoss()

        sgd_optimizer = torch.optim.SGD(resnet_pretrained_model.parameters(),
                                        lr=np.exp(orchestrate.io.assignment(Hyperparameters.LEARNING_RATE.value)),
                                        momentum=orchestrate.io.assignment(Hyperparameters.MOMENTUM.value),
                                        weight_decay=np.exp(
                                            orchestrate.io.assignment(Hyperparameters.WEIGHT_DECAY.value)),
                                        nesterov=orchestrate.io.assignment(Hyperparameters.NESTEROV.value))
        learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer, mode='min',
                                                                             factor=orchestrate.io.assignment(
                                                                                 Hyperparameters.LEARNING_RATE_SCHEDULER.value),
                                                                             patience=orchestrate.io.assignment(
                                                                                 Hyperparameters.SCEDULER_RATE.value),
                                                                             verbose=True)

        task = orchestrate.io.task
        num_epochs = Multitask[task.name].value
        logging.info("task assignment for multitask run: %s at cost %f", task.name, task.cost)
        logging.info("task corresponds to %d epochs", num_epochs)

        logging.info("training model")
        palm_net = PalmNet(epochs=num_epochs, gd_optimizer=sgd_optimizer, model=resnet_pretrained_model,
                           loss_function=cross_entropy_loss,
                           learning_rate_scheduler=learning_rate_scheduler,
                           validation_frequency=parsed_cli_arguments[CLI.VALIDATION_FREQUENCY.value],
                           torch_checkpoint_location=parsed_cli_arguments[CLI.CHECKPOINT.value],
                           model_checkpointing=parsed_cli_arguments[CLI.CHECKPOINT_FREQUENCY.value])

        trained_model, validation_metric = palm_net.train_model(training_data=DataLoader(training_data,
                                                                                         batch_size=2 ** (
                                                                                             orchestrate.io.assignment(
                                                                                                 Hyperparameters.BATCH_SIZE.value)),
                                                                                         shuffle=True),
                                                                validation_data=DataLoader(validation_data,
                                                                                           batch_size=2 ** (
                                                                                               orchestrate.io.assignment(
                                                                                                   Hyperparameters.BATCH_SIZE.value)),
                                                                                           shuffle=True),
                                                                number_of_labels=parsed_cli_arguments[
                                                                    CLI.NUM_CLASSES.value])

        return trained_model, validation_metric


if __name__ == "__main__":
    orchestrate_cli = OrchestrateCLI()
    orchestrate_cli.run_all()
