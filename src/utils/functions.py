from typing import List, Callable
import numpy
from scipy.stats import chi2
import chainer
import chainer.functions as F

from utils import UNK, EOS


def calculate_ln_hypervolume(ln_var: chainer.Variable, alpha: float) -> chainer.Variable:
    _, n_units = ln_var.shape
    xp = chainer.backends.cuda.get_array_module(ln_var)

    def ln_gamma(z):
        return xp.sum(xp.log(xp.arange(1, z)))

    ln_hv = xp.log(2) + (n_units * xp.log(xp.pi)) / 2 \
        - (xp.log(n_units) + ln_gamma(n_units / 2)) \
        + (n_units * xp.log(chi2.ppf(alpha, n_units))) / 2 \
        + F.sum(ln_var, axis=1)
    return ln_hv


def mask_unk(wy: numpy.ndarray) -> numpy.ndarray:
    xp = chainer.backends.cuda.get_array_module(wy)
    mask = xp.zeros_like(wy)
    mask[:, UNK] = -1024.0
    return wy + mask


def mask_eos(wy: numpy.ndarray) -> numpy.ndarray:
    xp = chainer.backends.cuda.get_array_module(wy)
    mask = xp.zeros_like(wy)
    mask[:, EOS] = -1024.0
    return wy + mask


def sequence_embed(embed: Callable, xs: List[numpy.ndarray]):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


def token_mask(xs: List[numpy.ndarray]) -> numpy.ndarray:
    xp = chainer.backends.cuda.get_array_module(*xs)
    batch = len(xs)
    max_length = max(len(x) for x in xs)

    mask = xp.ones((batch, max_length), 'f')
    for i, x in enumerate(xs):
        mask[i, len(x):] = 0
    return mask


def word_dropout(ys: List[numpy.ndarray], dropout_ratio: float) -> List[numpy.ndarray]:
    if not 0.0 <= dropout_ratio < 1.0:
        raise ValueError('dropout_ratio must be in the range [0, 1)')
    xp = chainer.backends.cuda.get_array_module(*ys)
    y_len = [len(y) for y in ys]
    y_section = numpy.cumsum(y_len[:-1])
    ys_concat = xp.concatenate(ys, axis=0)
    dropout_indices = xp.random.rand(*ys_concat.shape) <= dropout_ratio
    ys_concat[dropout_indices] = UNK
    ys_dropout = xp.split(ys_concat, y_section, 0)
    return ys_dropout
