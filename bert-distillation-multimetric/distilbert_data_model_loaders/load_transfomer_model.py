from transformers import DistilBertConfig, DistilBertForQuestionAnswering, BertConfig, \
    BertForQuestionAnswering, BertTokenizer, DistilBertTokenizer


MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
}


class LoadModel(object):

    def __init__(self, model_type):

        self.model_type = model_type

        assert self.model_type in MODEL_CLASSES.keys(), "model type has to be bert or distilbert"
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.model_type]

        self.config_class = config_class
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
