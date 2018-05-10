import tensorflow as tf

from estimator.Hook import Hook
from estimator.Estimator import Mode

class TensorBoardLogger(Hook):
    def __init__(self, dir='./logs'):
        self.dir = dir

    def after_build_model(self, estimator):
        metrics = estimator.metrics

        logs_metrics = metrics[Mode.EVAL]

        for name, var in logs_metrics.items():
            tf.summary.scalar(name, var)

        self.merged = tf.summary.merge_all()
        self.writers = { k: tf.summary.FileWriter(self.dir + '/' + k) for k in metrics.keys()}

    def after_run_batch(self, estimator,*args, **kwargs):
        summary = estimator.sess.run(self.merged)
        self.writers[estimator.mode].add_summary(summary, estimator.train_steps)

    def after_run_epoch(self, estimator, epoch, data, batch_size, tot_res):
        pass