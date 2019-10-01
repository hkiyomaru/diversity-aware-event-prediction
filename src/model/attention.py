"""Attention module."""
from typing import List, Tuple, Callable, Optional

import numpy
import chainer
import chainer.functions as F
import chainer.links as L


class AttentionWrapper(chainer.Chain):

    def __init__(self, n_units: int, decoder, att) -> None:
        super(AttentionWrapper, self).__init__()
        with self.init_scope():
            self.bos_state = chainer.Parameter(initializer=self.xp.random.randn(1, n_units).astype('f'))

        self.decoder = decoder
        self.att = att

        self.n_units = n_units

    def __call__(self,
                 hx: Optional[chainer.Variable],
                 cx: Optional[chainer.Variable],
                 eys: chainer.Variable,
                 x_mask: numpy.ndarray,
                 y_mask: Optional[numpy.ndarray],
                 hxs: Optional[chainer.Variable] = None
                 ) -> Tuple[chainer.Variable, chainer.Variable, List[chainer.Variable]]:
        batch, max_length, encoder_output_size = hxs.shape
        compute_context = None
        if self.att:
            compute_context = self.att(hxs, x_mask)
        h = F.broadcast_to(self.bos_state, (batch, self.n_units)) if hx is None else hx
        c = self.xp.zeros((batch, self.n_units), 'f') if cx is None else cx

        os, hs, cs = [], [], []
        for ey in F.separate(eys, axis=1):
            ey_ = ey
            if self.att is not None:
                context = compute_context(F.reshape(h, (batch, self.n_units)))
                ey_ = F.concat((ey_, context))
            c, h = self.decoder(c, h, ey_)
            os.append(h)
            hs.append(h)
            cs.append(c)
        batch_indices = self.xp.arange(0, batch)
        if y_mask is not None:
            last_indices = y_mask.sum(axis=1).astype('i') - 1
        else:
            last_indices = self.xp.full(batch, -1)
        os = F.transpose_sequence(os)
        if y_mask is not None:
            os = [o[:l+1] for o, l in zip(os, last_indices.tolist())]
        h = F.get_item(F.stack(hs, axis=1), (batch_indices, last_indices, Ellipsis))
        c = F.get_item(F.stack(cs, axis=1), (batch_indices, last_indices, Ellipsis))
        return h, c, os


class AttentionModule(chainer.Chain):

    def __init__(self, n_encoder_output_units: int, n_decoder_units: int, n_attention_units: int):
        super(AttentionModule, self).__init__()
        with self.init_scope():
            self.h = L.Linear(n_encoder_output_units, n_attention_units)
            self.s = L.Linear(n_decoder_units, n_attention_units)
            self.o = L.Linear(n_attention_units, 1)
        self.n_encoder_output_units = n_encoder_output_units
        self.n_attention_units = n_attention_units

    def __call__(self, hxs: chainer.Variable, x_mask: numpy.ndarray) -> Callable:
        """Returns a function that calculates context given decoder's state."""
        batch_size, max_length, encoder_output_size = hxs.shape
        encoder_factor = F.reshape(
            self.h(
                F.reshape(
                    hxs,
                    (batch_size * max_length, self.n_encoder_output_units)
                )
            ),
            (batch_size, max_length, self.n_attention_units)
        )
        mask = (1 - x_mask) * -1024.

        def compute_context(previous_state: chainer.Variable) -> chainer.Variable:
            decoder_factor = F.broadcast_to(
                F.reshape(
                    self.s(previous_state),
                    (batch_size, 1, self.n_attention_units)),
                (batch_size, max_length, self.n_attention_units))

            attention = F.softmax(
                F.reshape(
                    self.o(
                        F.reshape(
                            F.tanh(encoder_factor + decoder_factor),
                            (batch_size * max_length, self.n_attention_units)
                        )
                    ),
                    (batch_size, max_length)
                ) + mask
            )
            return F.reshape(F.matmul(attention[:, :, None], hxs, transa=True), (batch_size, encoder_output_size))

        return compute_context
