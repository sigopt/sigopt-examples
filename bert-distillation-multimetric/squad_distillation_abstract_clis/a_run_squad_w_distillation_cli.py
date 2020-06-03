import argparse

from distilbert_run_and_hpo_configurations.distilbert_squad_run_parameters import RunParameters, get_default_run_parameters
from distilbert_run_and_hpo_configurations.distilbert_squad_hpo_parameters import SGDHyperparameter, \
    get_default_hyperparameters, SquadArchitectureHyperparameter, ArchitectureHyperparameter


class ARunDistilBertSquadCLI(object):
    parser_prefix = "--"

    def __init__(self):
        pass

    def define_run_commandline_args(self):
        parser = argparse.ArgumentParser()
        squad_finetuning_run_defaults = get_default_run_parameters()

        parser.add_argument(
            self.parser_prefix + RunParameters.MODEL_TYPE.value,
            default="distilbert",
            type=str,
            required=False,
            help="Model type selected",
        )

        parser.add_argument(
            self.parser_prefix + RunParameters.MODEL_NAME_OR_PATH.value,
            default="distilbert-base-uncased",
            type=str,
            required=False,
            help="Path to pre-trained model or shortcut name selected",
        )

        # Distillation parameters (optional)
        parser.add_argument(
            self.parser_prefix + RunParameters.TEACHER_TYPE.value,
            default=None,
            type=str,
            help="Teacher type. Teacher tokenizer and student (model) tokenizer must output the same tokenization. Only for distillation.",
        )
        parser.add_argument(
            self.parser_prefix + RunParameters.TEACHER_NAME_OR_PATH.value,
            default=None,
            type=str,
            help="Path to the already SQuAD fine-tuned teacher model. Only for distillation.",
        )

        parser.add_argument(
            self.parser_prefix + RunParameters.OUTPUT_DIR.value,
            default=None,
            type=str,
            required=True,
            help="The output directory where the model checkpoints and predictions will be written.",
        )

        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            help="The input data dir. Should contain the .json files for the task."
                 + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
        )
        parser.add_argument(
            self.parser_prefix + RunParameters.TRAIN_FILE.value,
            default=None,
            type=str,
            help="The input training file. If a data dir is specified, will look for the file there"
                 + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
        )
        parser.add_argument(
            self.parser_prefix + RunParameters.PREDICT_FILE.value,
            default=None,
            type=str,
            help="The input evaluation file. If a data dir is specified, will look for the file there"
                 + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
        )

        parser.add_argument(
            self.parser_prefix + RunParameters.VERSION_2.value,
            type=lambda x: str(x).lower() == 'true',
            default=True,
            help="If true, the SQuAD examples contain some that do not have an answer.",
        )

        parser.add_argument(self.parser_prefix + RunParameters.LOAD_PRETRAINED_MODEL.value,
                            action="store_true",
                            help="if flag turned on, will use --model_type and --model_name_or_path to load "
                                 "pretrained model")

        parser.add_argument(self.parser_prefix + RunParameters.LOAD_SEMI_PRETRAINED_MODEL.value,
                            type=lambda x: str(x).lower() == 'true',
                            default=True,
                            help="if flag turned on, will use --model_type and --model_name_or_path to load "
                                 "model with pretrained weights")

        parser.add_argument(self.parser_prefix + RunParameters.DO_TRAIN.value,
                            type=lambda x: str(x).lower() == 'true',
                            default=True,
                            help="Whether to run training.")
        parser.add_argument(self.parser_prefix + RunParameters.DO_EVAL.value,
                            type=lambda x: str(x).lower() == 'true',
                            default=True,
                            help="Whether to run eval on the dev set.")
        parser.add_argument(
            self.parser_prefix + RunParameters.EVALUATE_DURING_TRAINING.value,
            type=lambda x: str(x).lower() == 'true',
            default=True,
            help="Rul evaluation during training at each logging step."
        )
        parser.add_argument(
            self.parser_prefix + RunParameters.DO_LOWER_CASE.value,
            type=lambda x: str(x).lower() == 'true',
            default=True,
            help="Set this flag if you are using an uncased model."
        )

        parser.add_argument(self.parser_prefix + RunParameters.NUM_TRAIN_EPOCHS.value,
                            default=squad_finetuning_run_defaults[
                                RunParameters.NUM_TRAIN_EPOCHS.value], type=float,
                            help="Total number of training epochs to perform."
                            )

        parser.add_argument(
            self.parser_prefix + RunParameters.N_BEST_SIZE.value,
            default=squad_finetuning_run_defaults[RunParameters.N_BEST_SIZE.value],
            type=int,
            help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
        )

        parser.add_argument(self.parser_prefix + RunParameters.LOGGING_STEPS.value, type=int,
                            default=squad_finetuning_run_defaults[
                                RunParameters.LOGGING_STEPS.value], help="Log every X updates steps.")
        parser.add_argument(self.parser_prefix + RunParameters.SAVE_STEPS.value, type=int,
                            default=squad_finetuning_run_defaults[
                                RunParameters.SAVE_STEPS.value], help="Save checkpoint every X updates steps.")
        parser.add_argument(
            self.parser_prefix + RunParameters.EVAL_ALL_CHECKPOINTS.value,
            action="store_true",
            help="Evaluate all checkpoints starting with the same prefix as "
                 "model_name ending and ending with step number",
        )
        parser.add_argument(self.parser_prefix + RunParameters.NO_CUDA.value, action="store_true",
                            help="Whether not to use CUDA when available")
        parser.add_argument(
            self.parser_prefix + RunParameters.OVERWRITE_OUTPUT_DIR.value, action="store_true",
            help="Overwrite the content of the output directory"
        )
        parser.add_argument(
            self.parser_prefix + RunParameters.OVERWRTIE_CACHE.value, action="store_true",
            help="Overwrite the cached training and evaluation sets"
        )
        parser.add_argument(self.parser_prefix + RunParameters.SEED.value, type=int,
                            default=squad_finetuning_run_defaults[
                                RunParameters.SEED.value], help="random seed for initialization")

        parser.add_argument(self.parser_prefix + RunParameters.LOCAL_RANK.value, type=int, default=-1,
                            help="local_rank for distributed training on gpus")
        parser.add_argument(
            self.parser_prefix + RunParameters.FP_16.value,
            action="store_true",
            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
        )
        parser.add_argument(
            "--fp16_opt_level",
            type=str,
            default="O1",
            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                 "See details at https://nvidia.github.io/apex/amp.html",
        )

        parser.add_argument(
            self.parser_prefix + RunParameters.VERBOSE_LOGGING.value,
            action="store_true",
            help="If true, all of the warnings related to data processing will be printed. "
                 "A number of warnings are expected for a normal SQuAD evaluation.",
        )

        parser.add_argument(
            self.parser_prefix + RunParameters.MAX_STEPS.value,
            default=-1,
            type=int,
            help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
        )

        parser.add_argument(
            self.parser_prefix + RunParameters.CACHE_DIR.value,
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )

        parser.add_argument(
            self.parser_prefix + RunParameters.TRAIN_CACHE_S3_DIRECTORY.value,
            type=str,
            help="location of squad2 cache files in s3. if present, will download cached features instead of loading "
                 "features"
        )

        parser.add_argument(
            self.parser_prefix + RunParameters.EVAL_CACHE_S3_DIRECTORY.value,
            type=str,
            help="location of squad2 cache files in s3. if present, will download cached features instead of loading "
                 "features"
        )

        parser.add_argument(
            self.parser_prefix + RunParameters.CACHE_S3_BUCKET.value,
            type=str,
            help="s3 bucket name for cache files"
        )

        # Other parameters

        parser.add_argument(
            self.parser_prefix + RunParameters.CONFIG_NAME.value, default="", type=str,
            help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            self.parser_prefix + RunParameters.TOKENIZER_NAME.value,
            default="",
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )

        return parser

    def define_common_hpo_commandline_args(self, parser):
        squad_finetuning_hpo_defaults = get_default_hyperparameters()

        parser.add_argument(
            self.parser_prefix + SGDHyperparameter.GRADIENT_ACCUMULATION_STEPS.value,
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )

        parser.add_argument(self.parser_prefix + SGDHyperparameter.MAX_GRAD_NORM.value,
                            default=squad_finetuning_hpo_defaults[
                                SGDHyperparameter.MAX_GRAD_NORM.value], type=float, help="Max gradient norm.")

        parser.add_argument(
            "--null_score_diff_threshold",
            type=float,
            default=0.0,
            help="If null_score - best_non_null is greater than the threshold predict null.",
        )

        # SQUAD preprocessing parameters

        parser.add_argument(
            self.parser_prefix + SquadArchitectureHyperparameter.MAX_SEQ_LENGTH.value,
            default=squad_finetuning_hpo_defaults[SquadArchitectureHyperparameter.MAX_SEQ_LENGTH.value],
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                 "longer than this will be truncated, and sequences shorter than this will be padded.",
        )
        parser.add_argument(
            self.parser_prefix + SquadArchitectureHyperparameter.DOC_STRIDE.value,
            default=squad_finetuning_hpo_defaults[SquadArchitectureHyperparameter.DOC_STRIDE.value],
            type=int,
            help="When splitting up a long document into chunks, how much stride to take between chunks.",
        )
        parser.add_argument(
            self.parser_prefix + SquadArchitectureHyperparameter.MAX_QUERY_LENGTH.value,
            default=squad_finetuning_hpo_defaults[SquadArchitectureHyperparameter.MAX_QUERY_LENGTH.value],
            type=int,
            help="The maximum number of tokens for the question. Questions longer than this will "
                 "be truncated to this length.",
        )
        parser.add_argument(
            self.parser_prefix + SquadArchitectureHyperparameter.MAX_ANSWER_LENGTH.value,
            default=squad_finetuning_hpo_defaults[SquadArchitectureHyperparameter.MAX_ANSWER_LENGTH.value],
            type=int,
            help="The maximum length of an answer that can be generated. This is needed because the start "
                 "and end predictions are not conditioned on one another.",
        )
        parser.add_argument(self.parser_prefix + ArchitectureHyperparameter.DIMENSION.value,
                            default=squad_finetuning_hpo_defaults[ArchitectureHyperparameter.DIMENSION.value],
                            type=int,
                            help="dimension of multihead attention, fully connected layer")
        parser.add_argument(self.parser_prefix + ArchitectureHyperparameter.HIDDEN_DIMENSION.value,
                            default=squad_finetuning_hpo_defaults[ArchitectureHyperparameter.HIDDEN_DIMENSION.value],
                            type=int, help="dim of hidden layers for linear layers")

        return parser

    def get_commandline_args(self):
        pass
