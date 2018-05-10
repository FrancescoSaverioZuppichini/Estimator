from data import input_fn
from data import data
from model import model_builder
from estimator.Estimator import Estimator
from estimator.Estimator import Mode
from estimator.TensorBoardLogger import TensorBoardLogger
from estimator.BasicLogger import BasicLogger

import numpy as np

EPOCHS = 200
BATCH_SIZE = 64

estimator = Estimator(model_builder, input_fn, hooks=[BasicLogger()])

# estimator.train(EPOCHS, data[Mode.TRAIN], batch_size=BATCH_SIZE)

estimator.train_and_evaluate(EPOCHS, data[Mode.TRAIN], validation=data[Mode.EVAL], batch_size=BATCH_SIZE)
# estimator.evaluate(data[Mode.EVAL])
print(estimator.predict(np.array([[1,2]])))

