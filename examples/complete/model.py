import tensorflow as tf
from estimator.Estimator import Mode

def model_builder(x, y, config):

    net = tf.layers.dense(x, 16, activation=tf.nn.relu)
    net = tf.layers.dense(net, 8, activation=tf.nn.relu)
    net = tf.layers.dense(net, 4, activation=tf.nn.relu)
    predictions = tf.layers.dense(net, 1, activation=tf.nn.sigmoid)

    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

    return {
            Mode.TRAIN: {'train_step': train_step },
            Mode.PREDICT: { 'predictions': predictions },
            Mode.EVAL: { 'loss': loss },
    }

