from estimator.estimator.Hook import Hook
from estimator.estimator.Estimator import  Mode
import numpy as np

class EarlyStopping(Hook):
    def __init__(self, *args, wait=5, after=0):
        super().__init__()
        self.last_mode  = Mode.PREDICT
        self.metric = [*args]
        self.history  = { Mode.EVAL : None, Mode.TRAIN: None}
        self.wait = wait
        self.after = after
        self.n = 0
        self.best_score = np.inf
        self.best_run = 0

    def compare(self, a, b):
        return a < b

    def get_metrics_results(self, tot_res):
        mean_metrics_res = { k: np.mean(tot_res[k]) for k in self.metric }

        return mean_metrics_res

    def update_stats(self, estimator):
        eval_score = np.array(list(self.history[Mode.EVAL].values()))

        if self.compare(eval_score, self.best_score):
            self.best_score = eval_score
            self.best_run = estimator.train_steps - 1

    def compare_val_train(self):
        train_res, eval_res = self.history[Mode.TRAIN], self.history[Mode.EVAL]

        return np.array(list(train_res.values())) > np.array(list(eval_res.values()))

    def should_stop(self, estimator):
        is_val_better = self.compare_val_train()

        return not is_val_better and estimator.train_steps > self.after

    def on_stop(self):
        print('Stopped after {} consecutive try'.format(self.wait))
        print('Best validation score on epoch {}'.format(self.best_run))

    def after_run_epoch(self, estimator, epoch, data, batch_n, tot_res):
        if estimator.mode != Mode.PREDICT:
            has_train_before_and_now_will_eval = self.last_mode ==  Mode.TRAIN and estimator.mode == Mode.EVAL

            self.history[estimator.mode] = self.get_metrics_results(tot_res)

            if estimator.mode == Mode.EVAL:
                self.update_stats(estimator)

            if has_train_before_and_now_will_eval:
                should_stop = self.should_stop(estimator)
                if should_stop:
                    self.n += 1
                else:
                    self.n = 0
                if self.n == self.wait:
                    estimator.stop()
                    self.on_stop()

            self.last_mode = estimator.mode
