import os
import torch
from transformers import SquadV2Processor, SquadV1Processor, squad_convert_examples_to_features
import boto3

from logger import logger
import logging
from distilbert_run_and_hpo_configurations.distilbert_squad_hpo_parameters import SquadArchitectureHyperparameter
from distilbert_run_and_hpo_configurations.distilbert_squad_run_parameters import RunParameters


def load_and_cache_examples(args_dict, tokenizer, evaluate=False, output_examples=False):
    if args_dict[RunParameters.LOCAL_RANK.value] not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_file = args_dict[RunParameters.PREDICT_FILE.value] if evaluate else args_dict[
        RunParameters.TRAIN_FILE.value]
    cached_features_file = os.path.join(
        os.path.dirname(input_file),
        "cached_distillation_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args_dict[RunParameters.MODEL_NAME_OR_PATH.value].split("/"))).pop(),
            str(args_dict[SquadArchitectureHyperparameter.MAX_SEQ_LENGTH.value]),
        ),
    )

    if os.path.exists(cached_features_file) is True:
        logging.info("deleting local cache file: {}".format(cached_features_file))
        os.remove(cached_features_file)

    download_cache_from_s3(args_dict, evaluate)

    if os.path.exists(cached_features_file) and args_dict[RunParameters.OVERWRTIE_CACHE.value] is False:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)

        try:
            features, dataset, examples = (
                features_and_dataset["features"],
                features_and_dataset["dataset"],
                features_and_dataset["examples"],
            )
        except KeyError:
            raise DeprecationWarning(
                "You seem to be loading features from an older version of this script please delete the "
                "file %s in order for it to be created again" % cached_features_file
            )
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        processor = SquadV2Processor() if args_dict[RunParameters.VERSION_2.value] else SquadV1Processor()
        if evaluate:
            examples = processor.get_dev_examples(args_dict["data_dir"],
                                                  filename=args_dict[RunParameters.PREDICT_FILE.value])
        else:
            examples = processor.get_train_examples(args_dict["data_dir"],
                                                    filename=args_dict[RunParameters.TRAIN_FILE.value])

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args_dict[SquadArchitectureHyperparameter.MAX_SEQ_LENGTH.value],
            doc_stride=args_dict[SquadArchitectureHyperparameter.DOC_STRIDE.value],
            max_query_length=args_dict[SquadArchitectureHyperparameter.MAX_QUERY_LENGTH.value],
            is_training=not evaluate,
            return_dataset="pt"
        )

        if args_dict[RunParameters.LOCAL_RANK.value] in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args_dict[RunParameters.LOCAL_RANK.value] == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def download_cache_from_s3(args_dict, evaluate):
    logging.info("Downloading cache from s3")
    s3_resource = boto3.resource('s3')
    cache_directory_destination = os.path.dirname(args_dict[RunParameters.TRAIN_FILE.value])
    if args_dict[RunParameters.TRAIN_CACHE_S3_DIRECTORY.value] is not None and evaluate is False:
        get_file_s3(s3_resource, args_dict[RunParameters.CACHE_S3_BUCKET.value], args_dict[
            RunParameters.TRAIN_CACHE_S3_DIRECTORY.value], cache_directory_destination)
    if args_dict[RunParameters.EVAL_CACHE_S3_DIRECTORY.value] is not None and evaluate is True:
        get_file_s3(s3_resource, args_dict[RunParameters.CACHE_S3_BUCKET.value], args_dict[
            RunParameters.EVAL_CACHE_S3_DIRECTORY.value], cache_directory_destination)


def get_file_s3(s3_resource, s3_bucket, s3_key, cache_dest):
    logging.info("downloading {}".format(s3_key))
    local_cache_path = os.path.join(cache_dest, os.path.basename(s3_key))
    s3_resource.Bucket(s3_bucket).download_file(
        s3_key,
        local_cache_path
    )
    assert os.path.exists(local_cache_path)

