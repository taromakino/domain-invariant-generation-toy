from enum import Enum


class Task(Enum):
    ALL = 'all'
    ERM_X = 'erm_x'
    ERM_C = 'erm_c'
    ERM_S = 'erm_s'
    VAE = 'vae'
    CLASSIFY = 'classify'


class EvalStage(Enum):
    TRAIN = 'train'
    VAL_ID = 'val_id'
    VAL_OOD = 'val_ood'
    TEST = 'test'