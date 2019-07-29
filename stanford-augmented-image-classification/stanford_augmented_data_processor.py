from a_stanford_data_processor import AStanfordDataProcessor
from stanford_car_dataset_augmented import StanfordCarAugmentedDataset
from stanford_car_dataset_augmented import preprocess_data
from stanford_cars_data_augmentation import StanfordCarsDataAugmentation


class StanfordAugmentedDataProcessor(AStanfordDataProcessor):

    def __init__(self, path_images, transforms, path_human_readable_labels):
        super().__init__(path_images, transforms, path_human_readable_labels)

    def augment_data(self,
                     path_to_data_mat,
                     augmentation_multiple,
                     s3_bucket_name,
                     store_to_disk,
                     store_to_s3,
                     probability,
                     brightness,
                     contrast,
                     saturation,
                     hue
                     ):
        stanford_cars_data_augmentation = StanfordCarsDataAugmentation(path_to_data_mat=path_to_data_mat,
                                                                       path_images=self.path_images)
        augmented_data_directory = stanford_cars_data_augmentation.augment_data(
            s3_bucket_name=s3_bucket_name,
            store_to_disk=store_to_disk,
            store_to_s3=store_to_s3,
            augmentation_multiple=augmentation_multiple,
            probability=probability,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue)
        return stanford_cars_data_augmentation.get_all_data_matrix(), augmented_data_directory

    def preprocess_data(self, augmented_mat_data, validation_percentage, data_subset):
        training_struct, validation_struct, unique_labels = preprocess_data(augmented_mat_data,
                                                                            validation_percentage,
                                                                            data_subset)
        return training_struct, validation_struct, unique_labels

    def get_data_generator(self, data_matrix):
        return StanfordCarAugmentedDataset(data_matrix=data_matrix,
                                           path_images=self.path_images,
                                           transforms=self.transforms,
                                           path_human_readable_labels=self.path_human_readable_labels)
