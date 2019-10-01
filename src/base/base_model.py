"""Basic seq2seq model."""
from typing import List

import numpy
import chainer


class BaseSeq2seq(chainer.Chain):

    def __init__(self) -> None:
        super(BaseSeq2seq, self).__init__()

    def __call__(self, *inp) -> chainer.Variable:
        raise NotImplementedError

    def translate(self, *inp) -> List[numpy.ndarray]:
        raise NotImplementedError
