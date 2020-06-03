from distilbert_data_model_loaders.load_transfomer_model import LoadModel
import logging


class LoadSemiPretrainedModel(LoadModel):

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

    def get_config(self, config_dict=dict()):
        # config_dict overwrites existing default values for given model_type
        return self.config_class.from_dict(config_dict)


def get_semi_pretrained_model(model_type, model_name_or_path, cache_dir, config_dict):
    logging.info("Loading semi-pretrained model with model type: {} and model path: {}".format(model_type, model_name_or_path))
    model_loader = LoadSemiPretrainedModel(model_type, model_name_or_path, cache_dir)
    config = model_loader.get_config(config_dict)
    model = model_loader.get_pretrained_model(config)
    return model_loader, model, config
