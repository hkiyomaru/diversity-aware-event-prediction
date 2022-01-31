"""Metrics to evaluate seq2seq models."""
from typing import List, Callable, Union
import collections

import numpy
from nltk.translate import bleu_score, nist_score
# import chainer


def get_bleu_precision_func(rs_list: List[List[List[int]]],
                            hs_list: List[List[List[int]]],
                            reduce: str = 'mean'
                            ) -> Callable:
    scores = []  # (N, H, R)
    for rs, hs in zip(rs_list, hs_list):
        hs_scores = []  # (H, R)
        for h in hs:
            h_scores = []  # (R,)
            for r in rs:
                bleu = bleu_score.sentence_bleu([r], h, smoothing_function=bleu_score.SmoothingFunction().method1)
                h_scores.append(bleu)
            hs_scores.append(h_scores)
        scores.append(hs_scores)

    def bleu_precision_func(at: int = 1, offset: int = 0) -> Union[float, List[float]]:
        scores_at = []  # (N,)
        for hs_scores in scores:
            assert len(hs_scores[offset:]) >= at, f'the length of hypotheses should be longer than {offset} + {at}'
            hs_scores_at = []  # (at,)
            for h_scores in hs_scores[offset:offset + at]:  # extract "at" hypotheses
                hs_scores_at.append(max(h_scores))
            scores_at.append(sum(hs_scores_at) / len(hs_scores_at))

        if reduce == 'mean':
            return sum(scores_at) / len(scores_at)
        elif reduce == 'no':
            return scores_at
        else:
            raise ValueError(f"only 'mean' and 'no' are valid for 'reduce', but '{reduce}' is given")

    return bleu_precision_func


def get_bleu_recall_func(rs_list: List[List[List[int]]],
                         hs_list: List[List[List[int]]],
                         reduce: str = 'mean'
                         ) -> Callable:
    scores = []  # (N, R, H)
    for rs, hs in zip(rs_list, hs_list):
        rs_scores = []  # (R, H)
        for r in rs:
            r_scores = []  # (H,)
            for h in hs:
                bleu = bleu_score.sentence_bleu([r], h, smoothing_function=bleu_score.SmoothingFunction().method1)
                r_scores.append(bleu)
            rs_scores.append(r_scores)
        scores.append(rs_scores)

    def bleu_recall_func(at: int = 1, offset: int = 0) -> Union[float, List[float]]:
        scores_at = []  # (N,)
        for rs_scores in scores:
            rs_scores_at = []  # (R,)
            for r_scores in rs_scores:
                assert len(r_scores[offset:]) >= at, f'the length of hypotheses should be longer than {offset} + {at}'
                rs_scores_at.append(max(r_scores[offset:offset + at]))
            scores_at.append(sum(rs_scores_at) / len(rs_scores_at))

        if reduce == 'mean':
            return sum(scores_at) / len(scores_at)
        elif reduce == 'no':
            return scores_at
        else:
            raise ValueError(f"only 'mean' and 'no' are valid for 'reduce', but '{reduce}' is given")

    return bleu_recall_func


def calculate_bleu(references, hypotheses):
    return bleu_score.corpus_bleu(references, hypotheses, smoothing_function=bleu_score.SmoothingFunction().method1)


def calculate_nist(references, hypotheses):
    return nist_score.corpus_nist(references, hypotheses)


def calculate_average_length(_, hypotheses):
    return sum([len(hypothesis) for hypothesis in hypotheses]) / len(hypotheses)


def calculate_distinct_1(_, hypotheses):
    return _calculate_distinct_n(hypotheses, 1)


def calculate_distinct_2(_, hypotheses):
    return _calculate_distinct_n(hypotheses, 2)


def _calculate_distinct_n(hypotheses, n):
    ngrams = [ngram(s, n) for s in hypotheses]
    n_ngrams = sum([len(e) for e in ngrams])
    unique_ngrams = [set(e) for e in ngrams]
    n_unique_ngrams = len(set.union(*unique_ngrams))
    return n_unique_ngrams / n_ngrams if n_ngrams > 0 else 0


def calculate_unigram_entropy(_, hypotheses, at: int = 1, offset: int = 0):
    target_hypotheses = []
    for hypotheses_ in hypotheses:
        assert len(hypotheses_[offset:offset + at]) >= at
        target_hypotheses.append(hypotheses_[offset:offset + at])
    return _calculate_entropy_n(target_hypotheses, 1)


def calculate_bigram_entropy(_, hypotheses, at: int = 1, offset: int = 0):
    target_hypotheses = []
    for hypotheses_ in hypotheses:
        assert len(hypotheses_[offset:offset + at]) >= at
        target_hypotheses.append(hypotheses_[offset:offset + at])
    return _calculate_entropy_n(target_hypotheses, 2)


def _calculate_entropy_n(hypotheses, n):
    ngrams = [_ngram for s in _flatten(hypotheses) for _ngram in ngram(s, n)]
    n_ngrams = len(ngrams)
    ngram_probabilities = numpy.array(list(collections.Counter(ngrams).values())) / n_ngrams
    entropy = numpy.average(-numpy.sum(ngram_probabilities * numpy.log(ngram_probabilities), axis=1))
    return entropy


def calculate_event_entropy(_, hypotheses, at: int = 1, offset: int = 0):
    target_hypotheses = []
    for hypotheses_ in hypotheses:
        assert len(hypotheses_[offset:offset + at]) >= at
        target_hypotheses.append(hypotheses_[offset:offset + at])

    unique_counts = []
    for hypotheses_ in target_hypotheses:
        hypotheses_str_ = ['-'.join(str(i) for i in hypothesis) for hypothesis in hypotheses_]
        unique_counts.append(list(collections.Counter(hypotheses_str_).values()))

    entropies = []
    for unique_count in unique_counts:
        unique_counts = numpy.array(unique_count)
        event_probablities = unique_counts / unique_counts.sum()
        entropies.append(-numpy.sum(event_probablities * numpy.log(event_probablities)))
    return sum(entropies) / len(entropies)


def ngram(s, n):
    return list(zip(*(s[i:] for i in range(n))))


def _flatten(l):
    return [e for sl in l for e in sl]


# class CalculateMetrics(chainer.training.Extension):
#
#     trigger = 1, 'epoch'
#     priority = chainer.training.PRIORITY_WRITER
#
#     def __init__(self, model, test_data, keys, functions, batch=100, device=-1, max_length=100):
#         self.model = model
#         self.test_data = test_data
#         self.keys = keys
#         self.functions = functions
#         self.batch = batch
#         self.device = device
#         self.max_length = max_length
#
#     def __call__(self, trainer):
#         with chainer.no_backprop_mode():
#             references = []
#             hypotheses = []
#             for i in range(0, len(self.test_data), self.batch):
#                 sources, targets = zip(*self.test_data[i:i + self.batch])
#                 references.extend([[t.tolist()] for t in targets])
#
#                 sources = [
#                     chainer.dataset.to_device(self.device, x) for x in sources]
#                 ys = [y.tolist()
#                       for y in self.model.translate(sources, self.max_length, sampling=False)]
#                 hypotheses.extend(ys)
#
#         for key, function in zip(self.keys, self.functions):
#             chainer.report({key: function(references, hypotheses)})
#
#
# class CalculateBleu(chainer.training.Extension):
#
#     trigger = 1, 'epoch'
#     priority = chainer.training.PRIORITY_WRITER
#
#     def __init__(
#             self, model, test_data, key, batch=100, device=-1, max_length=100):
#         self.model = model
#         self.test_data = test_data
#         self.key = key
#         self.batch = batch
#         self.device = device
#         self.max_length = max_length
#
#     def __call__(self, trainer):
#         with chainer.no_backprop_mode():
#             references = []
#             hypotheses = []
#             for i in range(0, len(self.test_data), self.batch):
#                 sources, targets = zip(*self.test_data[i:i + self.batch])
#                 references.extend([[t.tolist()] for t in targets])
#
#                 sources = [
#                     chainer.dataset.to_device(self.device, x) for x in sources]
#                 ys = [y.tolist()
#                       for y in self.model.translate(sources, self.max_length, sampling=False)]
#                 hypotheses.extend(ys)
#
#         bleu = bleu_score.corpus_bleu(
#             references, hypotheses,
#             smoothing_function=bleu_score.SmoothingFunction().method1)
#         chainer.report({self.key: bleu})
#
#
# class CalculateDistinctN(chainer.training.Extension):
#
#     trigger = 1, 'epoch'
#     priority = chainer.training.PRIORITY_WRITER
#
#     def __init__(
#             self, model, test_data, key, n=1, batch=100, device=-1, max_length=100):
#         self.model = model
#         self.test_data = test_data
#         self.key = key
#         self.n = n
#         self.batch = batch
#         self.device = device
#         self.max_length = max_length
#
#     def __call__(self, trainer):
#         with chainer.no_backprop_mode():
#             hypotheses = []
#             for i in range(0, len(self.test_data), self.batch):
#                 sources, _ = zip(*self.test_data[i:i + self.batch])
#
#                 sources = [
#                     chainer.dataset.to_device(self.device, x) for x in sources]
#                 ys = [y.tolist()
#                       for y in self.model.translate(sources, self.max_length, sampling=False)]
#                 hypotheses.extend(ys)
#
#         distinct_n = self.distinct_n(hypotheses)
#         chainer.report({self.key: distinct_n})
#
#     def distinct_n(self, hypotheses):
#         ngrams = [self.ngram(s, self.n) for s in hypotheses]
#         n_ngrams = sum([len(e) for e in ngrams])
#         unique_ngrams = [set(e) for e in ngrams]
#         n_unique_ngrams = len(set.union(*unique_ngrams))
#         return n_unique_ngrams / n_ngrams if n_ngrams > 0 else 0
#
#     @staticmethod
#     def ngram(s, n):
#         return list(zip(*(s[i:] for i in range(n))))
