from chainer.training.updaters import StandardUpdater


class StandardUpdaterWithParameter(StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.lambda_reconstruction = kwargs.pop('lambda_reconstruction', 1.0)
        super(StandardUpdaterWithParameter, self).__init__(*args, **kwargs)

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        additional_params = {
            'lambda_reconstruction': self.lambda_reconstruction
        }
        in_arrays.update(additional_params)

        if isinstance(in_arrays, tuple):
            optimizer.update(loss_func, *in_arrays)
        elif isinstance(in_arrays, dict):
            optimizer.update(loss_func, **in_arrays)
        else:
            optimizer.update(loss_func, in_arrays)
