import logging
import random

import numpy as np
import torch

from logger import logger
from distilbert_run_and_hpo_configurations.distilbert_squad_run_parameters import RunParameters

N_GPU = "n_gpu"
DEVICE = "device"


def set_seed(args_dict):
    logging.info("setting seed: {}".format(args_dict[RunParameters.SEED.value]))
    random.seed(args_dict[RunParameters.SEED.value])
    np.random.seed(args_dict[RunParameters.SEED.value])
    torch.manual_seed(args_dict[RunParameters.SEED.value])
    if args_dict[N_GPU] > 0:
        torch.cuda.manual_seed_all(args_dict[RunParameters.SEED.value])


def setup_dist_training(args_dict):
    logging.info("setting device")
    if args_dict[RunParameters.LOCAL_RANK.value] == -1 or args_dict[RunParameters.NO_CUDA.value]:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args_dict[RunParameters.NO_CUDA.value] else "cpu")
        logging.info("Device used is: {}".format(device))
        args_dict[N_GPU] = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args_dict[RunParameters.LOCAL_RANK.value])
        device = torch.device("cuda", args_dict[RunParameters.LOCAL_RANK.value])
        torch.distributed.init_process_group(backend="nccl")
        args_dict[N_GPU] = 1
    args_dict[DEVICE] = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args_dict[RunParameters.LOCAL_RANK.value],
        device,
        args_dict[N_GPU],
        bool(args_dict[RunParameters.LOCAL_RANK.value] != -1),
        args_dict[RunParameters.FP_16.value],
    )
    return args_dict
