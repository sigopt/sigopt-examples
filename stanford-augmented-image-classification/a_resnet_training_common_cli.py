import argparse
from torch.utils.data import DataLoader
from resnet import get_pretrained_resnet
import torch
from resnet import ResNet
import logging
from enum import Enum
import numpy as np
import torchvision


class CLI(Enum):
    DATA = 'path_data'
    CHECKPOINT = 'path_model_checkpoint'
    CHECKPOINT_FREQUENCY = 'checkpoint_frequency'
    EPOCHS = 'epochs'
    VALIDATION_FREQUENCY = 'validation_frequency'
    NUM_CLASSES = 'number_of_classes'
    DATA_SUBSET = 'data_subset'
    FREEZE_WEIGHTS = 'freeze_weights'
    IMAGES = 'path_images'
    LABELS = 'path_labels'
    MODEL = 'model'


class Hyperparameters(Enum):
    LEARNING_RATE_SCHEDULER = 'learning_rate_scheduler'
    BATCH_SIZE = 'batch_size'
    NESTEROV = 'nesterov'
    WEIGHT_DECAY = 'weight_decay'
    MOMENTUM = 'momentum'
    LEARNING_RATE = 'learning_rate'
    SCEDULER_RATE = 'scheduler_rate'


class AStanfordCarsCLI(object):

    IMAGE_TRANSFORMS = [torchvision.transforms.Resize(224),
                                                      torchvision.transforms.RandomCrop(224),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                       [0.229, 0.224, 0.225])]

    def __init__(self):
        pass

    def arg_parse(self):
        """CLI interface"""
        parser = argparse.ArgumentParser(description='CLI for tuning ResNet for Stanford Cars dataset.')
        parser.add_argument("--" + CLI.DATA.value, dest=CLI.DATA.value, type=str,
                            help="mat file with training and test data", required=True)
        parser.add_argument("--" + CLI.IMAGES.value, dest=CLI.IMAGES.value, type=str,
                            help="path to directory with images", required=True)
        parser.add_argument("--" + CLI.LABELS.value, dest=CLI.LABELS.value, type=str,
                            help="file with id and human readable label name", required=True)

        parser.add_argument("--" + CLI.CHECKPOINT.value, dest=CLI.CHECKPOINT.value, type=str,
                            help="directory to save model checkpoints", required=False, default=None)
        parser.add_argument("--" + CLI.MODEL.value, dest=CLI.MODEL.value, type=str,
                            help="model to use. options: ResNet18, ResNet50", required=True)
        parser.add_argument("--" + CLI.CHECKPOINT_FREQUENCY.value, dest=CLI.CHECKPOINT_FREQUENCY.value, type=int,
                            help="frequency to save model", required=False, default=None)
        parser.add_argument("--" + CLI.NUM_CLASSES.value, dest=CLI.NUM_CLASSES.value, type=int,
                            help='number of unique classes in labels', required=True)

        parser.add_argument("--" + CLI.EPOCHS.value, dest=CLI.EPOCHS.value, type=int,
                            help="total number of training epochs", required=True)
        parser.add_argument("--" + CLI.VALIDATION_FREQUENCY.value, dest=CLI.VALIDATION_FREQUENCY.value, type=int,
                            help="frequency to run validation", required=True)

        parser.add_argument("--" + CLI.DATA_SUBSET.value, dest=CLI.DATA_SUBSET.value, type=float,
                            help="subset of training data to use", required=True, default=1.0)

        parser.add_argument("--" + CLI.FREEZE_WEIGHTS.value, dest=CLI.FREEZE_WEIGHTS.value, action='store_true',
                            help="whether or not to freeze weights on pretrained model")

        parser.add_argument("--" + "no-" + CLI.FREEZE_WEIGHTS.value, dest=CLI.FREEZE_WEIGHTS.value,
                            action='store_false',
                            help="whether or not to freeze weights on pretrained model")

        return parser

    def load_datasets(self, *args):
        """Subclasses to implement dataset loading."""
        pass

    def get_run_arguments(self, parsed_cli_dict):
        """Subclasses to implement getting run arguments"""
        pass

    def run(self, parameter_arguments, num_epochs, training_data, validation_data):
        logging.info("loading pretrained model and establishing model characteristics")

        # get pretrained model from PyTorch model zoo
        resnet_pretrained_model = get_pretrained_resnet(is_freeze_weights=parameter_arguments[CLI.FREEZE_WEIGHTS.value],
                                                        number_of_labels=parameter_arguments[CLI.NUM_CLASSES.value],
                                                        model_type=parameter_arguments[CLI.MODEL.value])

        # define loss function
        cross_entropy_loss = torch.nn.CrossEntropyLoss()

        # define gradient descent strategy
        sgd_optimizer = torch.optim.SGD(resnet_pretrained_model.parameters(),
                                        lr=np.exp(parameter_arguments[Hyperparameters.LEARNING_RATE.value]),
                                        momentum=parameter_arguments[Hyperparameters.MOMENTUM.value],
                                        weight_decay=np.exp(
                                            parameter_arguments[Hyperparameters.WEIGHT_DECAY.value]),
                                        nesterov=parameter_arguments[Hyperparameters.NESTEROV.value])

        # define learning rate annealing
        learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer, mode='min',
                                                                             factor=parameter_arguments[
                                                                                 Hyperparameters.LEARNING_RATE_SCHEDULER.value],
                                                                             patience=parameter_arguments[
                                                                                 Hyperparameters.SCEDULER_RATE.value],
                                                                             verbose=True)

        logging.info("training model")
        resnet = ResNet(epochs=num_epochs, gd_optimizer=sgd_optimizer,
                        model=resnet_pretrained_model,
                        loss_function=cross_entropy_loss,
                        learning_rate_scheduler=learning_rate_scheduler,
                        validation_frequency=parameter_arguments[CLI.VALIDATION_FREQUENCY.value],
                        torch_checkpoint_location=parameter_arguments[CLI.CHECKPOINT.value],
                        model_checkpointing=parameter_arguments[CLI.CHECKPOINT_FREQUENCY.value])

        # train model
        trained_model, validation_metric = resnet.train_model(training_data=DataLoader(training_data, batch_size=2 ** (
            parameter_arguments[Hyperparameters.BATCH_SIZE.value]), shuffle=True),
                                                              validation_data=DataLoader(validation_data,
                                                                                         batch_size=2 ** (
                                                                                             parameter_arguments[
                                                                                                 Hyperparameters.BATCH_SIZE.value]),
                                                                                         shuffle=True),
                                                              number_of_labels=parameter_arguments[
                                                                  CLI.NUM_CLASSES.value])

        return trained_model, validation_metric

    def run_all(self):
        """Subclasses to implement driver for running CLI"""
        pass
