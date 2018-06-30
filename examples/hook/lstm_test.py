import tensorflow as tf
from tensorflow.contrib.framework import nest
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell
import numpy as np
tf.reset_default_graph()

batch_size = 64

inputs, targets = (np.random.rand(1000,2,1).astype(np.float32),np.random.rand(1000,1).astype(np.float32))

dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(batch_size).repeat()

x, y = dataset.make_one_shot_iterator().get_next()
# lstm = LSTMCell(2)
cell = MultiRNNCell([LSTMCell(2), LSTMCell(2)])

state = nest.map_structure(
    lambda x: tf.placeholder_with_default(x, x.shape, x.op.name),
    cell.zero_state(batch_size, tf.float32))

for tensor in nest.flatten(state):
    tf.add_to_collection('rnn_state_input', tensor)

out, new_state = tf.nn.dynamic_rnn(cell, x, initial_state=state)
out = tf.reshape(out, [-1, 1])
pred = tf.layers.Dense(units=1)(out)

loss = tf.losses.mean_squared_error(predictions=out, labels=y)
for tensor in nest.flatten(new_state):
    tf.add_to_collection('rnn_state_output', tensor)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(loss))