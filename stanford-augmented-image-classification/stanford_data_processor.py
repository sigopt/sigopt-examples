from a_stanford_data_processor import AStanfordDataProcessor
from stanford_car_dataset import StanfordCarDataset
from stanford_car_dataset import preprocess_data


class StanfordDataProcessor(AStanfordDataProcessor):

    def __init__(self, path_images, transforms, path_human_readable_labels):
        super().__init__(path_images, transforms, path_human_readable_labels)

    def preprocess_data(self, path_to_matdata, validation_percentage, data_subset):
        training_struct, validation_struct, unique_labels = preprocess_data(path_to_matdata,
                                                                                 validation_percentage,
                                                                          data_subset)
        return training_struct, validation_struct, unique_labels

    def get_data_generator(self, data_matrix):
        return StanfordCarDataset(data_matrix=data_matrix,
                                  path_images=self.path_images,
                                  transforms=self.transforms,
                                  path_human_readable_labels=self.path_human_readable_labels)
