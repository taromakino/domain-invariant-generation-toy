from enum import Enum


class Task(Enum):
    ERM_Y_C = 'erm_y_c'
    ERM_Y_X = 'erm_y_x'
    ERM_C_X = 'erm_c_x'
    TRAIN_VAE = 'train_vae'
    TRAIN_Q = 'train_q'
    INFERENCE = 'inference'