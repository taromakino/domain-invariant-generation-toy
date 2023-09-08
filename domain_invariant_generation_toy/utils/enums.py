from enum import Enum


class Task(Enum):
    ERM_Y_C = 'erm_y_c'
    ERM_Y_S = 'erm_y_s'
    ERM_Y_X = 'erm_y_x'
    TRAIN_VAE = 'train_vae'
    INFER_Z_TRAIN = 'infer_z_train'
    INFER_Z_VAL = 'infer_z_val'
    TRAIN_CLASSIFIER = 'train_classifier'
    INFERENCE = 'inference'