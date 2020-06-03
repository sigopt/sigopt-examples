from distilbert_data_model_loaders.load_transfomer_model import LoadModel
import logging


class LoadPretrainedModel(LoadModel):

    def __init__(self, model_type, model_name_or_path, cache_dir):
        super().__init__(model_type)
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir

    def get_pretrained_model(self, config):
        return self.model_class.from_pretrained(
            self.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_name_or_path),
            config=config,
            cache_dir=self.cache_dir if self.cache_dir else None,
        )

    def get_pretrained_config(self, config_name=None):
        return self.config_class.from_pretrained(
            config_name if config_name else self.model_name_or_path,
            cache_dir=self.cache_dir if self.cache_dir else None,
        )

    def get_tokenizer(self, max_positional_embedding_length, tokenizer_name=None, do_lower=True):
        return self.tokenizer_class.from_pretrained(
            tokenizer_name if tokenizer_name else self.model_name_or_path,
            do_lower_case=do_lower,
            cache_dir=self.cache_dir if self.cache_dir else None,
            max_len=max_positional_embedding_length,
        )


def get_pretrained_model(model_type, model_name_or_path, cache_dir):
    logging.info("loading pretrained model with model type: {}, model name or path: {}, and cache_dir: {}"
                 .format(model_type, model_name_or_path, cache_dir))
    pretrained_loader = LoadPretrainedModel(model_type=model_type,
                                            model_name_or_path=model_name_or_path,
                                            cache_dir=cache_dir)
    pretrained_config = pretrained_loader.get_pretrained_config()
    pretrained_model = pretrained_loader.get_pretrained_model(config=pretrained_config)
    return pretrained_loader, pretrained_model, pretrained_config


def get_pretrained_tokenizer(model_type, model_name_or_path, cache_dir, max_positional_embedding_length=512):
    logging.info("loading pretrained tokenizer with model type: {}, model name or path: {}, cache_dir: {}, and pos "
                 "embedding length: {}".format(model_type, model_name_or_path, cache_dir, max_positional_embedding_length))
    pretrained_loader = LoadPretrainedModel(model_type=model_type,
                                            model_name_or_path=model_name_or_path,
                                            cache_dir=cache_dir)
    pretrained_tokenizer = pretrained_loader.get_tokenizer(max_positional_embedding_length)
    return pretrained_tokenizer
