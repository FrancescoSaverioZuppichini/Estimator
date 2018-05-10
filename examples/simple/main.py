from estimator.Estimator import Estimator, Mode
import numpy as np
import tensorflow as tf

EPOCHS = 100
# create the dataset
train_data = (np.random.rand(1000,2),np.random.rand(1000,1))
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


estimator = Estimator(model_builder, input_fn)
# we can define a batch size before train, default is one
estimator.train(data=train_data, epochs=EPOCHS, batch_size=64)
res = estimator.evaluate(data=test_data)

print(res)
pred = estimator.predict(np.array([[2,1]]))
print(pred)
