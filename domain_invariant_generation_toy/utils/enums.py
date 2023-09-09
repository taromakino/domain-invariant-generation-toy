from enum import Enum


class Task(Enum):
    ERM_Y_C = 'erm_y_c'
    ERM_Y_S = 'erm_y_s'
    ERM_Y_X = 'erm_y_x'
    TRAIN_VAE = 'train_vae'
    TRAIN_ZC_SAMPLER = 'train_zc_sampler'
    INFERENCE = 'inference'