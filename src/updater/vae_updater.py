import numpy
from chainer.training.updaters import StandardUpdater


class StandardUpdaterForVAE(StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.k = kwargs.pop('num_sampling')
        self.c = kwargs.pop('cost')
        self.i = kwargs.pop('increase_rate')
        self.offset = kwargs.pop('offset')
        self.word_dropout = kwargs.pop('word_dropout')
        self.lambda_reconstruction = kwargs.pop('lambda_reconstruction', 1.0)
        super(StandardUpdaterForVAE, self).__init__(*args, **kwargs)

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        additional_params = {
            'n_sampling': self.k,
            'kl_cost': self.sigmoid((self.iteration + self.offset) * self.i) * self.c,
            'word_dropout_ratio': min(self.word_dropout, 0.05 * self.epoch),
            'lambda_reconstruction': self.lambda_reconstruction
        }
        in_arrays.update(additional_params)

        if isinstance(in_arrays, tuple):
            optimizer.update(loss_func, *in_arrays)
        elif isinstance(in_arrays, dict):
            optimizer.update(loss_func, **in_arrays)
        else:
            optimizer.update(loss_func, in_arrays)

    @staticmethod
    def sigmoid(x: float) -> float:
        x = numpy.clip(x, -500, 500)
        return 1.0 / (1.0 + numpy.exp(-x))
