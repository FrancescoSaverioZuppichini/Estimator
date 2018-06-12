class Hook:
    def before_build_model(self, estimator):
        pass

    def after_build_model(self, estimator):
        pass

    def before_run_epoch(self, estimator, epoch, data, batch_n, tot_res):
        pass

    def after_run_epoch(self, estimator, epoch, data, batch_n, tot_res):
        pass

    def before_run_batch(self, estimator, res, i):
        pass

    def after_run_batch(self, estimator, res, i, tot_res):
        pass
