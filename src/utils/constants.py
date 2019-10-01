"""Constant values."""
EOS = 0
UNK = 1
PAD = -1

CHECK_POINT = {
    'bleu': 'seq2seq.best_bleu.npz',
    'loss': 'seq2seq.best_loss.npz',
    'latest': 'seq2seq.latest.npz'
}
