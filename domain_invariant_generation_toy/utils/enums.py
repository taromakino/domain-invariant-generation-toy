from enum import Enum


class Task(Enum):
    ERM = 'erm'
    VAE = 'vae'
    Q_Z = 'q_z'
    CLASSIFY = 'classify'


class EvalStage(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'