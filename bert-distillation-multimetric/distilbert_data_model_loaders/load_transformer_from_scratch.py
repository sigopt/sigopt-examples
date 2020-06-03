from distilbert_data_model_loaders.load_transfomer_model import LoadModel
import logging


class LoadModelFromScratch(LoadModel):

    def __init__(self, model_type):
        super().__init__(model_type)

    def get_model(self, config):
        return self.model_class(config=config)

    def get_config(self, config_dict=dict()):
        # config_dict overwrites existing default values for given model_type
        return self.config_class.from_dict(config_dict)


def get_model_from_scratch(model_type, config_dict):
    model_loader = LoadModelFromScratch(model_type=model_type)
    config = model_loader.get_config(config_dict)
    model = model_loader.get_model(config)
    return model_loader, model, config
