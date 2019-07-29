from a_resnet_training_common_cli import Hyperparameters
from stanford_cars_augmentation_cli import AugmentHyperparameters, AugmentCLI


def generate_cli_hpo(parser):
    """Adding Hyperparameters to CLI arguments"""
    parser.add_argument("--" + Hyperparameters.SCEDULER_RATE.value, dest=Hyperparameters.SCEDULER_RATE.value,
                        type=float,
                        help="number of epochs to wait before annealing learning rate", required=True)
    parser.add_argument("--" + Hyperparameters.LEARNING_RATE.value, dest=Hyperparameters.LEARNING_RATE.value,
                        type=float,
                        help="learning rate to use", required=True)
    parser.add_argument("--" + Hyperparameters.BATCH_SIZE.value, dest=Hyperparameters.BATCH_SIZE.value, type=int,
                        help="batch size to use", required=True)
    parser.add_argument("--" + Hyperparameters.LEARNING_RATE_SCHEDULER.value,
                        dest=Hyperparameters.LEARNING_RATE_SCHEDULER.value, type=float,
                        help="annealing schedule rate to use. multiplied to learning rate", required=True)
    parser.add_argument("--" + Hyperparameters.WEIGHT_DECAY.value, dest=Hyperparameters.WEIGHT_DECAY.value, type=float,
                        help="weight decay to use", required=True)
    parser.add_argument("--" + Hyperparameters.MOMENTUM.value, dest=Hyperparameters.MOMENTUM.value, type=float,
                        help="momentum to use", required=True)
    parser.add_argument("--" + Hyperparameters.NESTEROV.value, dest=Hyperparameters.NESTEROV.value, action='store_true',
                        help="use Nesterov")
    parser.add_argument("--" + "no-" + Hyperparameters.NESTEROV.value, dest=Hyperparameters.NESTEROV.value,
                        action='store_false',
                        help="do not use Nesterov")
    return parser


def generate_cli_hpo_augment(parser):
    parser.add_argument("--" + AugmentHyperparameters.BRIGHTNESS.value, dest=AugmentHyperparameters.BRIGHTNESS.value,
                        type=float,
                        help="brightness factor. recommended range 0 - 9", required=True, default=3.2907)
    parser.add_argument("--" + AugmentHyperparameters.CONTRAST.value, dest=AugmentHyperparameters.CONTRAST.value,
                        type=float,
                        help="contrast factor. recommended range 0-100", required=True, default=56.793)
    parser.add_argument("--" + AugmentHyperparameters.HUE.value, dest=AugmentHyperparameters.HUE.value,
                        type=float,
                        help="hue factor. recommend range -0.5 - 0.5", required=True, default=-0.01286)
    parser.add_argument("--" + AugmentHyperparameters.SATURATION.value, dest=AugmentHyperparameters.SATURATION.value,
                        type=float,
                        help="saturation factor. recommended range 0-100", required=True, default=2.36640)
    return parser
