import tensorflow as tf
from estimator.Estimator import Mode
from estimator.Hook import Hook

class Saver(Hook):

    def __init__(self, save_dir):
        self.save_dir = save_dir

    def after_build_model(self, estimator):
        self.saver = tf.train.Saver()
        self.estimator = estimator

    def after_run_epoch(self, estimator, epoch, data, batch_size, tot_res):
        if estimator.mode == Mode.TRAIN:
            self.path = self.saver.save(estimator.sess, self.save_dir + '/' + estimator.name + '.ckpt', global_step=epoch)
            print('Saved mode in {}'.format(self.path))

    def restore(self, sess, path):
        self.saver.restore(sess, path)
        print('Restored from {}'.format(path))

    def restore_last(self, sess):
        path = tf.train.latest_checkpoint(self.save_dir)
        self.saver = tf.train.import_meta_graph(path + '.meta')
        self.restore(sess, tf.train.latest_checkpoint(self.save_dir))
