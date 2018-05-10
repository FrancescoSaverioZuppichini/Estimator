import tensorflow as tf
from tqdm import tqdm
import numpy as np
from estimator.Hook import Hook
from estimator.Estimator import Mode

class BasicLogger(Hook):
    def __init__(self):
        self.pbar = None
        self.n_digits = 0

    def before_run_epoch(self, estimator, epoch, data, batch_size, tot_res):
        n_batches = data[0].shape[0] // batch_size

        if len(str(n_batches)) > self.n_digits:
            self.n_digits = len(str(n_batches))

        if self.pbar:
            self.pbar.close()

        self.pbar = tqdm(range(n_batches))

    def format(self, estimator, res, i):
        eval_res = {k: res[k] for k in estimator.metrics[Mode.EVAL].keys()}

        return " ".join(["{}={:.4f}".format(k,v) for k,v in eval_res.items()])

    def after_run_batch(self, estimator, res, i,  tot_res):
        mean_res = {k: np.mean(tot_res[k]) for k in estimator.metrics[Mode.EVAL].keys()}
        # print(mean_res)
        self.pbar.set_description("Current: " + self.format(estimator,res, i) + ' AVG: ' + self.format(estimator,mean_res, i))
        self.pbar.update(1)

    def after_run_epoch(self, estimator, epoch, data, batch_size, tot_res):
    #     mean_tot_res = {k: np.mean(tot_res[k]) for k in estimator.metrics[Mode.EVAL].keys()}
    #     print("AVG Epoch {}: {}".format(epoch,self.format(estimator, mean_tot_res, epoch)))
        self.pbar.close()
