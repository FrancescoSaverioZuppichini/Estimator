from tqdm import tqdm
import numpy as np
from estimator.estimator.Hook import Hook
from estimator.estimator.Estimator import Mode

class BasicLogger(Hook):
    def __init__(self):
        self.pbar = None
        self.n_digits = 0
        self.epoch = 0

    def before_run_epoch(self, estimator, epoch, data, batch_size, tot_res):
        n_batches = data[0].shape[0] // batch_size

        if len(str(n_batches)) > self.n_digits:
            self.n_digits = len(str(n_batches))

        if self.pbar:
            self.pbar.close()

        self.pbar = tqdm(range(n_batches))

    def format(self, estimator, res, i):
        eval_res = {k: res[k] for k in estimator.get_operations(Mode.EVAL).keys()}

        return " ".join(["{}={:.4f}".format(k,v) for k,v in eval_res.items()])

    def after_run_batch(self, estimator, res, i,  tot_res):
        mean_res = {k: np.mean(tot_res[k]) for k in estimator.get_operations(Mode.EVAL).keys()}
        # print(mean_res)
        self.pbar.set_description("Epoch: " + str(self.epoch) + " Current: " + self.format(estimator,res, i) + ' AVG: ' + self.format(estimator,mean_res, i))
        self.pbar.update(1)

    def after_run_epoch(self, estimator, epoch, data, batch_size, tot_res):
        self.epoch = estimator.train_steps
    #     mean_tot_res = {k: np.mean(tot_res[k]) for k in estimator.metrics[Mode.EVAL].keys()}
    #     print("AVG Epoch {}: {}".format(epoch,self.format(estimator, mean_tot_res, epoch)))
        self.pbar.close()
