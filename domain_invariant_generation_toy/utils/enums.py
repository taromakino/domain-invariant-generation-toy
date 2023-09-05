from enum import Enum


class Task(Enum):
    ERM_Y_C = 'erm_y_c'
    ERM_Y_X = 'erm_y_x'
    ERM_C_X = 'erm_c_x'
    TRAIN_VAE = 'train_vae'
    TRAIN_Q = 'train_q'
    INFER_Z_TRAIN = 'infer_z_train'
    INFER_Z_VAL = 'infer_z_val'
    INFER_Z_TEST = 'infer_z_test'
    CLASSIFY_Y_ZC = 'classify_y_zc'
    CLASSIFY_C_ZC = 'classify_c_zc'
    REGRESS_S_ZC = 'regress_s_zc'