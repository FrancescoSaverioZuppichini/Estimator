from unittest import TestCase
from estimator.estimator.Estimator import Mode, Estimator
from estimator.estimator.EarlyStopping import EarlyStopping
from estimator.estimator.BasicLogger import BasicLogger

import tensorflow as tf
import numpy as np

train_data = (np.random.rand(100,2),np.random.rand(100,1))
test_data = (np.random.rand(100,2),np.random.rand(100,1))
# define the input function
input_fn = Estimator.create_input_fn(input_shape=[None,2],output_shape=[None,1])
# define the model builder
def model_builder(x, y, config):
    # config is a dictionary that can be passed to the estimator
    net = tf.layers.dense(x, 16, activation=tf.nn.relu)
    predictions = tf.layers.dense(net, 1, activation=tf.nn.sigmoid)

    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
    # it must return a dictionary contain the operation to train, predict and evaluate
    return {
            Mode.TRAIN: {'train_step': train_step },
            Mode.PREDICT: { 'predictions': predictions },
            Mode.EVAL: { 'loss': loss } # used as metrics
    }

class EarlyStoppingTest(TestCase):

    def setUp(self):
        self.early_stpping = EarlyStopping('loss',after=10)
        self.estimator = Estimator(model_builder=model_builder, input_fn=input_fn, hooks=[BasicLogger(), self.early_stpping])

    def test(self):
        self.estimator.train_and_evaluate(1000,  data=train_data, validation=test_data, batch_size=32, batch_size_eval=32)