from torch.utils.data import DataLoader
from resnet import get_pretrained_resnet
import torch
from resnet import PalmNet
import logging
from resnet_stanford_cars_cli import StanfordCarsCLI, Hyperparameters, CLI


class ResnetTrainingCLI(StanfordCarsCLI):
    def __init__(self):
        super().__init__()

    def arg_parse(self):
        """Adding Hyperparameters to CLI arguments"""
        parser = super().arg_parse()
        parser.add_argument("--" + Hyperparameters.SCEDULER_RATE.value, dest=Hyperparameters.SCEDULER_RATE.value, type=float,
                            help="number of epochs to wait before annealing learning rate", required=True)
        parser.add_argument("--" + Hyperparameters.LEARNING_RATE.value, dest=Hyperparameters.LEARNING_RATE.value, type=float,
                            help="learning rate to use", required=True)
        parser.add_argument("--" + Hyperparameters.BATCH_SIZE.value, dest=Hyperparameters.BATCH_SIZE.value, type=int,
                            help="batch size to use", required=True)
        parser.add_argument("--" + Hyperparameters.LEARNING_RATE_SCHEDULER.value, dest=Hyperparameters.LEARNING_RATE_SCHEDULER.value, type=float,
                            help="annealing schedule rate to use. multiplied to learning rate", required=True)
        parser.add_argument("--" + Hyperparameters.WEIGHT_DECAY.value, dest=Hyperparameters.WEIGHT_DECAY.value, type=float,
                            help = "weight decay to use", required=True)
        parser.add_argument("--" + Hyperparameters.MOMENTUM.value, dest=Hyperparameters.MOMENTUM.value, type=float,
                            help = "momentum to use", required=True)
        parser.add_argument("--" + Hyperparameters.NESTEROV.value, dest=Hyperparameters.NESTEROV.value, action='store_true',
                            help="use Nesterov")
        parser.add_argument("--" + "no-"+ Hyperparameters.NESTEROV.value, dest=Hyperparameters.NESTEROV.value, action='store_false',
                            help="do not use Nesterov")

        return parser

    def load_datasets(self, parsed_cli_arguments):
        return super().load_datasets(parsed_cli_arguments)

    def run(self, parsed_cli_arguments, training_data, validation_data):

        logging.info("loading pretrained model and establishing model characteristics")

        resnet_pretrained_model = get_pretrained_resnet(parsed_cli_arguments[CLI.FREEZE_WEIGHTS.value],
                                                        parsed_cli_arguments[CLI.NUM_CLASSES.value],
                                                        parsed_cli_arguments[CLI.MODEL.value])
        cross_entropy_loss = torch.nn.CrossEntropyLoss()

        sgd_optimizer = torch.optim.SGD(resnet_pretrained_model.parameters(),
                                        lr=parsed_cli_arguments[Hyperparameters.LEARNING_RATE.value],
                                        momentum=parsed_cli_arguments[Hyperparameters.MOMENTUM.value],
                                        weight_decay=parsed_cli_arguments[Hyperparameters.WEIGHT_DECAY.value],
                                        nesterov=parsed_cli_arguments[Hyperparameters.NESTEROV.value])
        learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer, mode='min',
                                                                             factor=parsed_cli_arguments[Hyperparameters.LEARNING_RATE_SCHEDULER.value],
                                                                             patience=parsed_cli_arguments[Hyperparameters.SCEDULER_RATE.value],
                                                                             verbose=True)

        logging.info("training model")
        palm_net = PalmNet(epochs=parsed_cli_arguments[CLI.EPOCHS.value], gd_optimizer=sgd_optimizer, model=resnet_pretrained_model,
                           loss_function=cross_entropy_loss,
                           learning_rate_scheduler=learning_rate_scheduler,
                           validation_frequency=parsed_cli_arguments[CLI.VALIDATION_FREQUENCY.value],
                           torch_checkpoint_location=parsed_cli_arguments[CLI.CHECKPOINT.value],
                           model_checkpointing=parsed_cli_arguments[CLI.CHECKPOINT_FREQUENCY.value])

        trained_model, validation_metric = palm_net.train_model(training_data=DataLoader(training_data,
                                                                                         batch_size=parsed_cli_arguments[Hyperparameters.BATCH_SIZE.value],
                                                                                         shuffle=True),
                                                                validation_data=DataLoader(validation_data,
                                                                                           batch_size=parsed_cli_arguments[Hyperparameters.BATCH_SIZE.value],
                                                                                           shuffle=True),
                                                                number_of_labels=parsed_cli_arguments[
                                                                    CLI.NUM_CLASSES.value])
        return trained_model, validation_metric


if __name__ == "__main__":
    resnet_training_cli = ResnetTrainingCLI()
    resnet_training_cli.run_all()
