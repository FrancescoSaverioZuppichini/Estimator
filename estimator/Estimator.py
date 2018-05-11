import tensorflow as tf
from tqdm import tqdm

class Mode:
    TRAIN = 'train'
    VAL = 'validation',
    TEST = 'test'
    EVAL = 'eval'
    PREDICT = 'predict'
    LOGS = 'logs'

# class Input:
#     TRAIN = 'train'
#     VAL = 'validation'
#     TEST = 'test'

SUMMARIES_DIR = 'logs'

class Logger:
    pass

class EstimatorConfig:

    def __init__(self, batch_size=None,shuffle=True):
        pass

class Estimator:
    def __init__(self, model_builder, input_fn, config=None, hooks=None, restore=False, name='model'):
        self.model_builder = model_builder
        self.input_fn = input_fn
        self.config = config
        self.name = name
        self.mode = None
        self.sess = tf.InteractiveSession()
        self.built = False
        # TODO decide what to do -> allow to use a dataset from numpy or not
        self.batch_size = tf.placeholder(tf.int64, name='batch_size')
        self.data_sets, self.x_, self.y_ = input_fn(self.batch_size)
        self.x, self.y = self.make_iter()

        self.hooks = hooks
        if not restore:
            self.build_model(self.x, self.y)
        if not restore:
            self.sess.run(tf.global_variables_initializer())
        self.train_steps = 0

    def make_iter(self,):
        # TODO check that TRAIN exist
        iter = tf.data.Iterator.from_structure(self.data_sets[Mode.TRAIN].output_types,
                                               self.data_sets[Mode.TRAIN].output_shapes)

        self.iter_init_ops = { k: iter.make_initializer(v, name='iter_init_{}'.format(k)) for k, v in self.data_sets.items()}

        return iter.get_next(name='iter_next')


    def build_model(self, x, y):
        if not self.built:
            self.metrics = self.model_builder(x, y, self.config)

            for hook in self.hooks:
                hook.after_build_model(self)

        self.built = True

    @property
    def operations(self):
        ops = self.metrics[self.mode]
        if self.mode == Mode.TRAIN:
            ops = { **ops, **self.metrics[Mode.EVAL]} # add eval to train

        return ops

    def restore(self):
        pass

    def run_epoch(self, sess, epoch, data, batch_size, n_batches=1):
        i = 0
        ops = self.operations

        tot_res = {k: [] for k in ops.keys()}
        res = {}

        for hook in self.hooks:
            hook.before_run_epoch(self, epoch, data, batch_size, tot_res)

        for batch_n in  range(n_batches):

                for hook in self.hooks:
                    hook.before_run_batch(self, res, i)

                res = sess.run(ops)

                for k,v in res.items():
                    tot_res[k].append(v)

                for hook in self.hooks:
                    hook.after_run_batch(self, res, i, tot_res)

                i += 1
                self.train_steps += 1

        for hook in self.hooks:
            hook.after_run_epoch(self, epoch, data, batch_size, tot_res)

        return tot_res

    def train(self, epochs, data, batch_size=1):

        self.mode = Mode.TRAIN
        self.data = data

        len_data = len(data[0])
        n_batches = len_data // batch_size

        self.sess.run(self.iter_init_ops[self.mode], feed_dict={ self.x_ : data[0], self.y_: data[1], 'batch_size:0': batch_size })

        for epoch in range(epochs):
            self.run_epoch(self.sess, epoch, data, batch_size, n_batches)

    def train_and_evaluate(self, epochs, data, validation, batch_size, batch_size_eval=None):
        # todo add param to specify every epoch to evaluate
        for epoch in range(epochs):
            self.train(epochs=1, data = data, batch_size=batch_size)
            self.evaluate(data=validation, batch_size=batch_size_eval)

    def evaluate(self, data, batch_size=None):
        self.mode = Mode.EVAL
        self.data = data

        batch_size = data[0].shape[0] if batch_size == None else batch_size
        n_batches = data[0].shape[0] // batch_size

        self.sess.run(self.iter_init_ops[self.mode], feed_dict={ self.x_ : data[0], self.y_: data[1], 'batch_size:0': batch_size})

        tot_res = self.run_epoch(self.sess, 0, data, batch_size, n_batches)

        return tot_res

    def predict(self, input):
        self.mode = Mode.PREDICT

        return self.sess.run(self.operations, feed_dict={ self.x : input })

    @staticmethod
    def create_input_fn(input_shape, output_shape, buffer_size=10000):

        def input_fn(batch_size):
            x, y = tf.placeholder(tf.float32, shape=input_shape), tf.placeholder(tf.float32, shape=output_shape)

            return { Mode.TRAIN: tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size).repeat().batch(batch_size),
                    Mode.EVAL: tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()}, x, y

        return input_fn

