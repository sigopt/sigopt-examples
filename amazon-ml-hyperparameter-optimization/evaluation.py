import base64
import boto3
import os
import logging
import config
import time

logger = logging.getLogger(config.APP_NAME)


class Evaluation(object):
    def __init__(self, fold, model_spec):
        self.fold = fold
        self.model_spec = model_spec
        self.ml_id = None
        self.ml_name = None
        self.ev_id = None
        self.ev_name = None
        self.auc = None

    def build(self):
        self._ml = boto3.client('machinelearning', region_name='us-east-1')
        self.create_ml_model()
        self.create_eval()

    def cleanup(self):
        self._ml.delete_evaluation(EvaluationId=self.ev_id)
        logger.info("Deleted Evaluation " + self.ev_id)
        self._ml.delete_ml_model(MLModelId=self.ml_id)
        logger.info("Deleted ML Model " + self.ml_id)

    def __str__(self):
        """
        Returns the string representing this fold object. The string
        includes the IDs of entities newly created on Amazon ML.
        """
        return """\n\
Evalaution {fold_ordinal} of {kfolds}:
 - Evaluation ID: {ev_id}
 - ML Model ID: {ml_id}""".format(
    fold_ordinal=self.fold.fold_ordinal,
    kfolds=self.fold.kfolds,
    **self.__dict__
)

    def create_ml_model(self):
        """
        Creates ML Model on Amazon ML using the training datasource.
        """
        self.ml_id = "ml-" + base64.b32encode(os.urandom(10)).decode("ascii")
        self.ml_name = "ML model: " + self.fold.train_ds_name
        self._ml.create_ml_model(
            MLModelId=self.ml_id,
            MLModelName=self.ml_name,
            TrainingDataSourceId=self.fold.train_ds_id,
            MLModelType=self.model_spec.ml_model_type,
            Parameters={
                "sgd.maxPasses": self.model_spec.sgd_maxPasses,
                "sgd.maxMLModelSizeInBytes": self.model_spec.sgd_maxMLModelSizeInBytes,
                self.model_spec.sgd_RegularizationType: str(self.model_spec.sgd_RegularizationAmount),
            },
            Recipe=self.model_spec.recipe,
        )
        logger.info("Created ML Model " + self.ml_id)

    def create_eval(self):
        """
        Created Evaluation on Amazon ML using the evaluation datasource.
        """
        self.ev_id = "ev-" + base64.b32encode(os.urandom(10)).decode("ascii")
        self.ev_name = "Evaluation: " + self.ml_name
        self._ml.create_evaluation(
            EvaluationId=self.ev_id,
            EvaluationName=self.ev_name,
            MLModelId=self.ml_id,
            EvaluationDataSourceId=self.fold.eval_ds_id
        )
        logger.info("Created Evaluation " + self.ev_id)

    @staticmethod
    def poll_eval(cls):
        start_timestamp = time.time()  # start timestamp in seconds

        # time delay in seconds between two polling attempt
        polling_delay = config.INITIAL_POLLING_DELAY

        logger.info("Checking the Evaluation status for %s...", cls.ev_id)
        while time.time() - start_timestamp < config.TIME_OUT:
            evaluation = cls._ml.get_evaluation(EvaluationId=cls.ev_id)
            eval_status = evaluation["Status"]
            logger.info("{} status: {}".format(cls.ev_id, eval_status))
            if eval_status == "COMPLETED":
                break
            elif eval_status == "FAILED":
                raise Exception("Evaluation {} is FAILED!".format(cls.ev_id))

            time.sleep(polling_delay)
            # update polling_delay in the next polling
            polling_delay = min(polling_delay * 2, config.DELAY_CAP)
            logger.debug("Next poll in {} seconds...".format(polling_delay))

        auc = evaluation["PerformanceMetrics"]["Properties"]["BinaryAUC"]
        cls.auc = float(auc)
