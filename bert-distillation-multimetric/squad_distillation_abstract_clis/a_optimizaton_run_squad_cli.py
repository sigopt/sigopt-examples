from squad_distillation_abstract_clis.a_run_squad_w_distillation_cli import ARunDistilBertSquadCLI
from distilbert_run_and_hpo_configurations.distilbert_squad_run_parameters import OptimizationRunParameters
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class OptimizeDistilBertQuadCLI(ARunDistilBertSquadCLI):

    def __init__(self):
        super().__init__()

    def define_run_commandline_args(self):
        parser = super().define_run_commandline_args()
        parser = super().define_common_hpo_commandline_args(parser)

        parser.add_argument(super().parser_prefix + OptimizationRunParameters.EXPERIMENT_NAME.value, type=str,
                            required=False,
                            help="SigOpt "
                                 "experiment name")
        parser.add_argument(super().parser_prefix + OptimizationRunParameters.PROJECT_NAME.value, type=str,
                            required=False,
                            help="SigOpt project name")
        parser.add_argument(super().parser_prefix + OptimizationRunParameters.USE_HPO_DEFAULT_RANGES.value,
                            action="store_true",
                            help="If flagged, "
                                 "will use default "
                                 "hyperparameter values.")
        parser.add_argument(super().parser_prefix + OptimizationRunParameters.API_TOKEN.value, type=str, required=True,
                            help="SigOpt API Token to use")
        parser.add_argument(super().parser_prefix + OptimizationRunParameters.API_URL.value, type=str, required=True,
                            help="SigOpt API URL to use")

        parser.add_argument(super().parser_prefix + OptimizationRunParameters.SIGOPT_EXPERIMENT_ID.value, type=str,
                            required=False, default=None,
                            help="If provided, will load SigOpt experiment of given ID.")

        parser.add_argument(super().parser_prefix + OptimizationRunParameters.STORE_S3.value, action="store_true",
                            help="Flag to store runs on s3")

        parser.add_argument(super().parser_prefix + OptimizationRunParameters.S3_BUCKET.value, type=str,
                            required=False,
                            help="s3 bucket name to "
                                 "store run outputs.")
        parser.add_argument(super().parser_prefix + OptimizationRunParameters.SIGOPT_OBSERVATION_BUDGET.value, type=int,
                            required=True,
                            help="Number of observations for SigOpt")
        return parser

    def get_commandline_args(self):
        parser = self.define_run_commandline_args()
        return parser
