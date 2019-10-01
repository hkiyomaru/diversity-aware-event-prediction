"""Usual seq2seq model."""
from typing import List, Tuple, Optional

import numpy
import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L

from base import BaseSeq2seq
from model import AttentionWrapper, AttentionModule
from utils import sequence_embed, mask_unk, mask_eos, token_mask
from utils import EOS


class Seq2seq(BaseSeq2seq):

    def __init__(self,
                 n_layers: int,
                 n_source_vocab: int,
                 n_target_vocab: int,
                 n_source_embed: int,
                 n_target_embed: int,
                 n_units: int,
                 attention: bool = False,
                 reconstruction: bool = False,
                 reverse: bool = False,
                 source_initialW: Optional[numpy.ndarray] = None,
                 target_initialW: Optional[numpy.ndarray] = None,
                 ) -> None:
        super(Seq2seq, self).__init__()

        self.n_layers = n_layers
        self.n_source_vocab = n_source_vocab + 2
        self.n_target_vocab = n_target_vocab + 2

        self.n_encoder_input_units = n_source_embed
        self.n_encoder_hidden_units = n_units * 2
        self.n_encoder_output_units = n_units * 2

        self.n_context_units = self.n_encoder_output_units if attention else 0
        self.n_decoder_input_units = n_target_embed + self.n_context_units
        self.n_decoder_hidden_units = self.n_encoder_hidden_units
        self.n_decoder_output_units = self.n_decoder_hidden_units

        self.n_context_units_rc = self.n_decoder_output_units if attention else 0
        self.n_decoder_input_units_rc = n_source_embed + self.n_context_units_rc
        self.n_decoder_hidden_units_rc = self.n_decoder_hidden_units
        self.n_decoder_output_units_rc = self.n_decoder_hidden_units_rc

        self.attention = attention
        self.reconstruction = reconstruction
        self.reverse = reverse

        with self.init_scope():
            self.embed_x = L.EmbedID(self.n_source_vocab, n_source_embed, initialW=source_initialW)
            self.embed_y = L.EmbedID(self.n_target_vocab, n_target_embed, initialW=target_initialW)

            self.encoder = L.NStepBiLSTM(
                self.n_layers, self.n_encoder_input_units, self.n_encoder_hidden_units // 2, 0.5)  # Because of BiLSTM

            self.decoder = L.StatelessLSTM(self.n_decoder_input_units, self.n_decoder_hidden_units)
            self.maxout = L.Maxout(self.n_decoder_output_units, n_units, 2)
            self.w = L.Linear(n_units, self.n_target_vocab)

            if self.attention:
                self.att = AttentionModule(self.n_encoder_output_units, self.n_decoder_hidden_units, n_units)
            else:
                self.att = None
            self._decoder = AttentionWrapper(self.n_decoder_hidden_units, self.decoder, self.att)

            if self.reconstruction:
                self.decoder_rc = L.StatelessLSTM(self.n_decoder_input_units_rc, self.n_decoder_hidden_units_rc)
                self.maxout_rc = L.Maxout(self.n_decoder_output_units_rc, n_units, 2)
                self.w_rc = L.Linear(n_units, self.n_source_vocab)

                if self.attention:
                    self.att_rc = AttentionModule(self.n_decoder_output_units, self.n_decoder_hidden_units_rc, n_units)
                else:
                    self.att_rc = None
                self._decoder_rc = AttentionWrapper(self.n_decoder_hidden_units_rc, self.decoder_rc, self.att_rc)

    def __call__(self,
                 xs: List[numpy.ndarray],
                 ys: List[numpy.ndarray],
                 lambda_reconstruction: float = 1.0
                 ) -> chainer.Variable:
        batch = len(xs)
        if self.reverse:
            ys = [y[::-1] for y in ys]
        eos = self.xp.array([EOS], numpy.int32)
        ys_in = [self.xp.concatenate([eos, y], axis=0) for y in ys]
        ys_out = [self.xp.concatenate([y, eos], axis=0) for y in ys]
        x_mask = token_mask(xs)
        y_mask = token_mask(ys_out)

        # encode input sequences
        hx, enc_os = self.encode(xs)

        # decode sequences from the input representations
        hx = None if self.attention else hx
        eys = F.pad_sequence(sequence_embed(self.embed_y, ys_in), padding=0.)
        enc_os = F.pad_sequence(enc_os, padding=0.)
        hy, _, dec_os = self._decoder(hx, None, eys, x_mask, y_mask, enc_os)

        # calculate the loss
        concat_os = F.concat(dec_os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.sum(F.softmax_cross_entropy(
            self.w(self.maxout(concat_os)), concat_ys_out, reduce='no')) / batch
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.data * batch / n_words)

        if self.reconstruction:
            # prepare input and output for backward prediction
            if self.reverse:
                xs = [x[::-1] for x in xs]
            xs_in = [F.concat([eos, x], axis=0) for x in xs]
            xs_out = [F.concat([x, eos], axis=0) for x in xs]
            x_mask_rc = token_mask(ys_out)
            y_mask_rc = token_mask(xs_out)

            # decode sequences from the input representations
            hy = None if self.attention else hy
            exs = F.pad_sequence(sequence_embed(self.embed_x, xs_in), padding=0.)
            dec_os = F.pad_sequence(dec_os, padding=0.)
            _, _, os_rc = self._decoder_rc(hy, None, exs, x_mask_rc, y_mask_rc, dec_os)

            # calculate the loss
            concat_os_rc = F.concat(os_rc, axis=0)
            concat_xs_out = F.concat(xs_out, axis=0)
            loss_rc = F.sum(F.softmax_cross_entropy(
                self.w_rc(self.maxout_rc(concat_os_rc)), concat_xs_out, reduce='no')) / batch
            loss += lambda_reconstruction * loss_rc

        chainer.report({'loss': loss.data, 'perp': perp}, self)
        return loss

    def translate(self,
                  xs: List[numpy.ndarray],
                  max_length: int = 25,
                  sampling: bool = True
                  ) -> List[numpy.ndarray]:
        batch = len(xs)
        x_mask = token_mask(xs)
        y_mask = None

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            # encode input sequences
            hx, enc_os = self.encode(xs)

            # decode sequences from the input representations
            ys = self.xp.full(batch, EOS, numpy.int32)
            h = None if self.attention else hx
            c = None
            enc_os = F.pad_sequence(enc_os, padding=0.)
            result = []
            for i in range(max_length):
                # encode current target words
                eys = self.embed_y(ys)[:, None, :]

                h, c, os = self._decoder(h, c, eys, x_mask, y_mask, enc_os)

                concat_os = F.concat(os, axis=0)
                wy = self.w(self.maxout(concat_os)).data
                wy = mask_unk(wy)  # forbid to output UNK
                if i == 0:
                    wy = mask_eos(wy)  # forbid to output EOS at first

                if sampling:
                    ys = chainer.distributions.Categorical(F.softmax(wy)).sample().data
                else:
                    ys = self.xp.argmax(wy, axis=1).astype(numpy.int32)
                result.append(ys)
            result = cuda.to_cpu(self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)

        outs = []
        for y in result:
            inds = numpy.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        if self.reverse:
            outs = [out[::-1] for out in outs]
        return outs

    def encode(self, xs: List[numpy.ndarray]) -> Tuple[chainer.Variable, List[chainer.Variable]]:
        batch = len(xs)
        exs = sequence_embed(self.embed_x, xs)
        hx, _, os = self.encoder(None, None, exs)
        hx = hx[-2:]  # extract last hidden states
        hx = F.reshape(F.swapaxes(hx, 0, 1), (batch, self.n_encoder_output_units))
        return hx, os
