
class AStanfordDataProcessor(object):

    def __init__(self, path_images, transforms, path_human_readable_labels):
        self.path_images = path_images
        self.transforms = transforms
        self.path_human_readable_labels = path_human_readable_labels

    def preprocess_data(self, path_to_matdata, validation_percentage, data_subset):
        pass

    def get_data_generator(self, data_matrix):
        pass
