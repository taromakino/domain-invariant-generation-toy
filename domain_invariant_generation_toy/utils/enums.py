from enum import Enum


class Task(Enum):
    ERM_Y_C = 'erm_y_c'
    ERM_Y_S = 'erm_y_s'
    ERM_Y_X = 'erm_y_x'
    VAE = 'vae'
    AGG_POSTERIOR = 'agg_posterior'
    INFER_Z = 'infer_z'
    CLASSIFY = 'classify'

class InferenceStage(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'