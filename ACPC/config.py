from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainParams:
    TRAIN_VAL_TEST_SPLIT = [0.8, 0.1, 0.1]
    SPLIT_SEED = 42
    TRAINING_EPOCH = 30
    TRAIN_CUDA = True
    BCE_WEIGHTS = [0.004, 0.996]
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    BACKGROUND_AS_CLASS = False
    LEARNING_RATE = 0.001

@dataclass
class EvaluateParams:
    TRAIN_CUDA = True
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    BACKGROUND_AS_CLASS = False

@dataclass
class DatasetParams:
    DATASET_PATH = '/home/richards/MonaiData/'
    TASK_ID = 9
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    TRAIN_BATCH_SIZE = 1
    VAL_BATCH_SIZE = 1
    TEST_BATCH_SIZE = 1