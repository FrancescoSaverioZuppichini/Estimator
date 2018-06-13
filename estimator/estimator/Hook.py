class Hook:
    def before_build_model(self, estimator):
        """
        This function is called before calling the `model_fn` function
        :param estimator: A reference to the estimator
        :return:
        """
        pass

    def after_build_model(self, estimator):
        """
        This function is called after calling the `model_fn` function
        :param estimator: A reference to the estimator
        :return:
        """
        pass

    def before_run_epoch(self, estimator, epoch, data, batch_n, tot_res):
        """
        This function is called before run a new epoch.
        :param estimator: A reference to the estimator
        :param epoch: The index of the current epoch
        :param data: The current data passed to the epoch
        :param batch_n: The number of batches
        :param tot_res: An object with keys the current mode operations and values and array of results
        :return:
        """
        pass

    def after_run_epoch(self, estimator, epoch, data, batch_n, tot_res):
        """
        This function is called after run a new epoch. It can be implemented in order to create
        custom hooks. For example, `tot_res` is a dictionary that contains all the results for each operation for each batch.
        For example we may want to log the epoch history during train.
        ```
        def after_run_epoch(self, estimator, epoch, data, batch_n, tot_res):
            if estimator.mode == Mode.TRAIN:
                # you can check the estimator mode
                print(tot_res) # { 'loss' : [0.8, 0.7, 0. 6]...}
        ```
        :param estimator: A reference to the estimator
        :param epoch: The index of the current epoch
        :param data: The current data passed to the epoch
        :param batch_n: The number of batches
        :param tot_res: An object with keys the current mode operations and values and array of results
        :return:
        """
        pass

    def before_run_batch(self, estimator, res, i):
        """
        This function is called before run a new batch.
        :param estimator: A reference to the estimator
        :param res: An object with keys the current mode operations and the single result for each one of them
        :param i: The batch index
        :return:
        """
        pass

    def after_run_batch(self, estimator, res, i, tot_res):
        """
        This function is called after run a new batch.
        :param estimator: A reference to the estimator
        :param res: An object with keys the current mode operations and the single result for each one of them
        :param i: The batch index
        :return:
        """
        pass
