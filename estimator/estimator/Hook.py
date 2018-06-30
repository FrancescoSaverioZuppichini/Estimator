class Hook:
    def before_build_model(self, estimator):
        """
        This function is called before calling the `model_fn` function

        Args:
            estimator: A reference to the estimator

        Returns:

        """
        pass

    def after_build_model(self, estimator):
        """
        This function is called after calling the `model_fn` function
        Args:
            estimator: A reference to the estimator

        Returns:

        """
        pass

    def before_run_epoch(self, estimator, epoch, data, batch_n, tot_res):
        """
        This function is called before run a new epoch.
        Args:
            estimator: A reference to the estimator
            epoch: The index of the current epoch
            data: The current data passed to the epoch
            batch_n: The number of batches
            tot_res: An object with keys the current mode operations and values and array of results

        Returns:

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

        Args:
            estimator: A reference to the estimator
            epoch: The index of the current epoch
            data: The current data passed to the epoch
            batch_n: The number of batches
            tot_res: An object with keys the current mode operations and values and array of results

        Returns:

        """
        pass

    def before_run_batch(self, estimator, res, i):
        """
        This function is called before run a new batch.

        Args:
            estimator: A reference to the estimator
            res: An object with keys the current mode operations and the single result for each one of them
            i:  The batch index

        Returns:

        """
        pass

    def after_run_batch(self, estimator, res, i, tot_res):
        """
        This function is called after run a new batch.

        Args:
            estimator: A reference to the estimator
            res: An object with keys the current mode operations and the single result for each one of them
            i:  The batch index

        Returns:

        """
        pass

