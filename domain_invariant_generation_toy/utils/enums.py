from enum import Enum


class Task(Enum):
    ERM_Y_C = 'erm_y_c'
    ERM_Y_S = 'erm_y_s'
    ERM_Y_X = 'erm_y_x'
    VAE = 'vae'
    Q_Z = 'q_z'
    CLASSIFY_TRAIN = 'classify_train'
    CLASSIFY_TEST = 'classify_test'