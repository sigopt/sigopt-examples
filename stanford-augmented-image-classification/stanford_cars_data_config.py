from enum import Enum
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class CarCommonDatasetAttributes(object):
    LABEL = 'class'
    REL_IMAGE_PATH = 'relative_im_path'
    ANNOTATIONS = 'annotations'


class CarDatasetAttributes(CarCommonDatasetAttributes):
    BBOX_X1 = 'bbox_x1'
    BBOX_X2 = 'bbox_x2'
    BBOX_Y1 = 'bbox_y1'
    BBOX_Y2 = 'bbox_y2'
    TEST = 'test'


class CarAugmentedDatasetAttributes(CarCommonDatasetAttributes):
    IS_AUGMENT = 'is_augment'
