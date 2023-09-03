from enum import Enum


class Task(Enum):
    ERM = 'erm'
    TRAIN_VAE = 'train_vae'
    TRAIN_INFERENCE_ENCODER = 'train_inference_encoder'
    CLASSIFY = 'classify'