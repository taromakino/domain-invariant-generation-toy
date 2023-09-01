from enum import Enum


class Task(Enum):
    ERM = 'erm'
    TRAIN_VAE = 'train_vae'
    TRAIN_Q = 'train_q'
    INFER_Z_TRAIN = 'infer_z_train'
    INFER_Z_VAL = 'infer_z_val'
    INFER_Z_TEST = 'infer_z_test'
    CLASSIFY = 'classify'