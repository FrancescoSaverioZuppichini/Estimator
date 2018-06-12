import tensorflow as tf

from estimator.estimator.Hook import Hook
from estimator.estimator.Estimator import Mode

class TensorBoardLogger(Hook):
    def __init__(self, dir='./logs'):
        self.dir = dir
        self.batch_i = {}
        self.epoch_i = {}

    def after_build_model(self, estimator):
        metrics = estimator.metrics

        logs_metrics = metrics[Mode.EVAL]

        for name, var in logs_metrics.items():
            tf.summary.scalar(name, var)

        self.merged = tf.summary.merge_all()
        self.writers = { k: tf.summary.FileWriter(self.dir + '/' + k) for k in metrics.keys()}
        self.batch_i = { k: 0  for k in metrics.keys()}

    def after_run_batch(self, estimator, res, i, *args, **kwargs):
        summary = estimator.sess.run(self.merged)
        self.writers[estimator.mode].add_summary(summary, self.batch_i[estimator.mode])
        self.batch_i[estimator.mode] += 1

    def after_run_epoch(self, estimator, epoch, data, batch_size, tot_res):
        pass