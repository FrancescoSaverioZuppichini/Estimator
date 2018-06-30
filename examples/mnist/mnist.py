from estimator.estimator.Estimator import Estimator, Mode
from estimator.estimator.BasicLogger import BasicLogger
from estimator.estimator.EarlyStopping import EarlyStopping
from estimator.estimator.TensorBoardLogger import TensorBoardLogger

import numpy as np
import tensorflow as tf
from keras.datasets import mnist

N_CLASSES = 10

(train_x, train_y), (test_x, test_y) = mnist.load_data()

# we need to add a dimension to the targets
train_y = np.expand_dims(train_y, -1)
test_y = np.expand_dims(test_y, -1)

tf.set_random_seed(0)

train_data = (train_x, train_y)
test_data = (test_x, test_y)

shape = train_data[0].shape[1:]

input_fn = Estimator.create_input_fn([None, *shape], [None, 1], output_type=tf.uint8)

def model(x,y, config):
    x = tf.reshape(x, [-1, 28*28])
    net = tf.layers.dense(x, 256, activation=tf.nn.relu)
    net = tf.layers.dropout(net, 0.2)
    net = tf.layers.dense(net, 128, activation=tf.nn.relu)
    out = tf.layers.dense(net, N_CLASSES)
    predictions = tf.nn.softmax(out)

    y_one_hot = tf.one_hot(y, N_CLASSES, dtype=tf.float32)
    y_one_hot = tf.reshape(y_one_hot, [-1, N_CLASSES])

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_one_hot, logits=out))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(y_one_hot, 1)), tf.float32))

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    # it must return a dictionary contain the operation to train, predict and evaluate
    return {
            Mode.TRAIN: {'train_step': train_step },
            Mode.PREDICT: { 'predictions': predictions },
            Mode.EVAL: { 'loss': loss, 'accuracy' : accuracy }
    }


estimator = Estimator(model_fn=model, input_fn=input_fn, hooks=[TensorBoardLogger(),BasicLogger(), EarlyStopping('loss',wait=2)])

# estimator.train(data=train_data, epochs=100, batch_size=64)

estimator.train_and_evaluate(data=train_data, validation=test_data, epochs=100, batch_size=64)