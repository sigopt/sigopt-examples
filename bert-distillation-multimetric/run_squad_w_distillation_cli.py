import squad_fine_tuning.squad_w_distillation
from squad_distillation_abstract_clis.a_run_squad_w_distillation_cli import ARunDistilBertSquadCLI
from distilbert_run_and_hpo_configurations.distilbert_squad_hpo_parameters import get_default_hyperparameters, DistillationHyperparameter, \
    SGDHyperparameter, ArchitectureHyperparameter


class RunDistilBertSquadCLI(ARunDistilBertSquadCLI):

    def __init__(self):
        super().__init__()

    def define_hpo_commandline_args(self, parser):

        squad_finetuning_hpo_defaults = get_default_hyperparameters()

        # Distillation hyperparameters

        parser.add_argument(
            self.parser_prefix + DistillationHyperparameter.ALPHA_CE.value, default=squad_finetuning_hpo_defaults[
                DistillationHyperparameter.ALPHA_CE.value], type=float,
            help="Distillation loss linear weight. Only for distillation."
        )
        parser.add_argument(
            self.parser_prefix + DistillationHyperparameter.ALPHA_SQUAD.value, default=squad_finetuning_hpo_defaults[
                DistillationHyperparameter.ALPHA_SQUAD.value], type=float,
            help="True SQuAD loss linear weight. Only for distillation."
        )
        parser.add_argument(
            self.parser_prefix + DistillationHyperparameter.TEMPERATURE.value, default=squad_finetuning_hpo_defaults[
                DistillationHyperparameter.TEMPERATURE.value], type=float,
            help="Distillation temperature. Only for distillation."
        )

        # SGD Hyperparameter

        parser.add_argument(self.parser_prefix + SGDHyperparameter.PER_COMPUTE_TRAIN_BATCH_SIZE.value,
                            default=squad_finetuning_hpo_defaults[SGDHyperparameter.PER_COMPUTE_TRAIN_BATCH_SIZE.value],
                            type=int,
                            help="Batch size per GPU/CPU for training.")
        parser.add_argument(
            self.parser_prefix + SGDHyperparameter.PER_COMPUTE_EVAL_BATCH_SIZE.value,
            default=squad_finetuning_hpo_defaults[
                SGDHyperparameter.PER_COMPUTE_EVAL_BATCH_SIZE.value], type=int,
            help="Batch size per GPU/CPU for evaluation."
        )
        parser.add_argument(self.parser_prefix + SGDHyperparameter.LEARNING_RATE.value,
                            default=squad_finetuning_hpo_defaults[
                                SGDHyperparameter.LEARNING_RATE.value], type=float,
                            help="The initial learning rate for Adam.")

        parser.add_argument(self.parser_prefix + SGDHyperparameter.WEIGHT_DECAY.value,
                            default=squad_finetuning_hpo_defaults[
                                SGDHyperparameter.WEIGHT_DECAY.value], type=float,
                            help="Weight decay if we apply some.")
        parser.add_argument(self.parser_prefix + SGDHyperparameter.ADAM_EPSILON.value,
                            default=squad_finetuning_hpo_defaults[SGDHyperparameter.ADAM_EPSILON.value],
                            type=float, help="Epsilon for Adam optimizer.")

        parser.add_argument(self.parser_prefix + SGDHyperparameter.WARM_UP_STEPS.value,
                            default=squad_finetuning_hpo_defaults[
                                SGDHyperparameter.WARM_UP_STEPS.value], type=int,
                            help="Linear warmup over warmup_steps.")

        # Architecture Parameters
        parser.add_argument(self.parser_prefix + ArchitectureHyperparameter.N_HEADS.value,
                            default=squad_finetuning_hpo_defaults[ArchitectureHyperparameter.N_HEADS.value],
                            type=int, help="number of attn heads")
        parser.add_argument(self.parser_prefix + ArchitectureHyperparameter.N_LAYERS.value,
                            default=squad_finetuning_hpo_defaults[ArchitectureHyperparameter.N_LAYERS.value],
                            type=int, help="number of attn heads")
        parser.add_argument(self.parser_prefix + ArchitectureHyperparameter.DROPOUT.value,
                            default=squad_finetuning_hpo_defaults[ArchitectureHyperparameter.DROPOUT.value],
                            type=int, help="number of attn heads")
        parser.add_argument(self.parser_prefix + ArchitectureHyperparameter.ATTENTION_DROPOUT.value,
                            default=squad_finetuning_hpo_defaults[ArchitectureHyperparameter.ATTENTION_DROPOUT.value],
                            type=int, help="number of attn heads")
        parser.add_argument(self.parser_prefix + ArchitectureHyperparameter.QA_DROPOUT.value,
                            default=squad_finetuning_hpo_defaults[ArchitectureHyperparameter.QA_DROPOUT.value],
                            type=int, help="number of attn heads")
        parser.add_argument(self.parser_prefix + ArchitectureHyperparameter.INIT_RANGE.value,
                            default=squad_finetuning_hpo_defaults[ArchitectureHyperparameter.INIT_RANGE.value],
                            type=int, help="number of attn heads")

        return parser

    def get_commandline_args(self):
        parser = self.define_run_commandline_args()
        parser = self.define_common_hpo_commandline_args(parser)
        parser = self.define_hpo_commandline_args(parser)
        return parser


if __name__ == "__main__":
    cli = RunDistilBertSquadCLI()
    parser = cli.get_commandline_args()
    args = parser.parse_args()
    args_dict = vars(args)
    model, results, eval_time = squad_fine_tuning.squad_w_distillation.main(args_dict)
