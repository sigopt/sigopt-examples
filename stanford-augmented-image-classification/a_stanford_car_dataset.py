from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset
from stanford_cars_data_config import CarCommonDatasetAttributes
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class AStanfordCarDataset(Dataset):

    TRANSFORMED_IMAGE = 'train'
    LABEL = 'label'

    def __init__(self, data_matrix, path_images, transforms, path_human_readable_labels):
        self.data_matrix = data_matrix
        self.transforms = transforms
        self.path_images = path_images
        self.path_human_readable_labels = path_human_readable_labels
        self.human_readable_labels = None
        self.load_human_readable_labels()

    def load_human_readable_labels(self):
        self.human_readable_labels = loadmat(self.path_human_readable_labels)['class_names']

    def get_label_unique_count(self):
        return np.unique(self.data_matrix[CarCommonDatasetAttributes.LABEL], return_counts=True)

    def get_class_distribution(self):
        return self.get_label_unique_count()[1]/len(self.data_matrix)

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
