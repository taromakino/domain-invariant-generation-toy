from enum import Enum


class Task(Enum):
    ERM = 'erm'
    TRAIN_VAE = 'train_vae'
    TRAIN_INFERENCE_ENCODER = 'train_inference_encoder'
    CLASSIFY_Y_ZC = 'class_y_zc'
    CLASSIFY_C_ZC = 'classify_c_zc'
    REGRESS_S_ZC = 'regress_s_zc'