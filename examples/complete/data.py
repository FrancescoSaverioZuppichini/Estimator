import tensorflow as tf
import numpy as np
from estimator.Estimator import Mode
from estimator.Estimator import Estimator

# train, test = tf.keras.datasets.mnist.load_data()

DATA_SIZE = [100000,200,150]

data = { Mode.TRAIN : (np.random.rand(DATA_SIZE[0],2), np.random.rand(DATA_SIZE[0],1)),
         Mode.VAL: (np.random.rand(DATA_SIZE[1],2), np.random.rand(DATA_SIZE[1],1)),
         Mode.EVAL: (np.random.rand(DATA_SIZE[1], 2), np.random.rand(DATA_SIZE[1], 1)) }


input_fn = Estimator.create_input_fn([None,2], [None,1])

# def input_fn(batch_size):
#     x, y = tf.placeholder(tf.float32, shape=[None, 2]), tf.placeholder(tf.float32, shape=[None, 1])
#
#     return { Mode.TRAIN: tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000).repeat().batch(batch_size),
#              Mode.EVAL: tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat() }, x, y