"""Evaluate BART model."""
import pathlib
from typing import List, Tuple, Dict
import argparse
import os
import collections

import numpy
import pandas
import torch
import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

from utils.utils import load_vocabulary
from utils.metrics import get_bleu_precision_func
from utils.metrics import get_bleu_recall_func
from utils.constants import UNK


seed = 42
numpy.random.seed(seed)
torch.manual_seed(seed)


def load_data(path: str, debug: bool = False) -> List[str]:
    raw = []
    with open(path) as f:
        for i, line in enumerate(f):
            raw.append(line.strip())
            if debug and i == 10:
                break
    return raw


def to_ids(s: str, vocab: Dict[str, int]) -> numpy.ndarray:
    return numpy.array([vocab.get(w, UNK) for w in s.split()], numpy.int32)


def aggregate_data(sources: List[str], targets: List[str]) -> Tuple[List[str], List[List[str]]]:
    source2pairs = collections.defaultdict(list)
    for source, target in zip(sources, targets):
        source2pairs[source].append(target)
    sources_, targets_ = [], []
    for s, ts in source2pairs.items():
        sources_.append(s)
        targets_.append(ts)
    return sources_, targets_


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate test results in a latex format')
    parser.add_argument('--model_name_or_path', nargs='*', help='path pattern to result directories')
    parser.add_argument('--data_dir', help='path to data directory')
    parser.add_argument('--num-example', default=15, type=int, help='number of sampling')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--debug', action='store_true', help='whether to use small dataset')
    parser.add_argument('--use-original-test-data', action='store_true', help='whether to use original test set')
    parser.add_argument('--batchsize', default=64, type=int, help='the number of batchsize')
    parser.add_argument('--max_source_length', default=64, type=int, help='max source length')
    parser.add_argument('--max_target_length', default=64, type=int, help='max target length')
    args = parser.parse_args()

    # load configuration files with sorting metrics
    model_paths = [pathlib.Path(path) for path in args.model_name_or_path]

    word_ids = load_vocabulary(os.path.join(args.data_dir, 'vocab.txt'))

    # load dataset
    if args.use_original_test_data:
        test_source_path = os.path.join(args.data_dir, 'test.source')
        test_target_path = os.path.join(args.data_dir, 'test.target')
    else:
        test_source_path = os.path.join(args.data_dir, 'mturk.source')
        test_target_path = os.path.join(args.data_dir, 'mturk.target')
    test_source = load_data(test_source_path, debug=args.debug)
    test_target = load_data(test_target_path, debug=args.debug)
    assert len(test_source) == len(test_target)

    if args.use_original_test_data:
        test_source, test_target = test_source, [[t] for t in test_target]
    else:
        test_source, test_target = aggregate_data(test_source, test_target)

    n_repeat = 1 if args.use_original_test_data else args.num_example
    test_source_ = []
    for test_source_i in test_source:
        test_source_.extend([test_source_i] * n_repeat)

    n_loop = -(-len(test_source_) // args.batchsize)

    device = 'cpu' if args.gpu < 0 else f'cuda:{args.gpu}'

    results = []
    for model_path in model_paths:
        model = BartForConditionalGeneration.from_pretrained(model_path)
        model.to(device)
        tokenizer = BartTokenizer.from_pretrained(model_path)

        generated = []
        for i in tqdm.trange(n_loop):
            batch_source = test_source_[i * args.batchsize:(i + 1) * args.batchsize]
            inputs = tokenizer(
                batch_source,
                max_length=args.max_source_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            if args.use_original_test_data:
                kwargs = {'do_sample': False, 'num_beams': 1}
            else:
                kwargs = {'do_sample': True}

            outputs = model.generate(**inputs, **kwargs)
            generated.extend(tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ))

        references = [[to_ids(s, word_ids) for s in ss] for ss in test_target]
        hypotheses = [to_ids(s, word_ids) for s in generated]
        hypotheses = [hypotheses[i * n_repeat: (i + 1) * n_repeat] for i in range(len(test_source))]

        if args.use_original_test_data:
            bleu = get_bleu_precision_func(references, hypotheses)(at=1, offset=0) * 100
            result = {'summary': 'bart', 'bleu': bleu}
        else:
            bleu_precision_func = get_bleu_precision_func(references, hypotheses)
            bleu_precision_5 = bleu_precision_func(at=5, offset=0) * 100
            bleu_precision_10 = bleu_precision_func(at=10, offset=5) * 100

            bleu_recall_func = get_bleu_recall_func(references, hypotheses)
            bleu_recall_5 = bleu_recall_func(at=5, offset=0) * 100
            bleu_recall_10 = bleu_recall_func(at=10, offset=5) * 100

            bleu_f_5 = (2 * bleu_precision_5 * bleu_recall_5) / (bleu_precision_5 + bleu_recall_5)
            bleu_f_10 = (2 * bleu_precision_10 * bleu_recall_10) / (bleu_precision_10 + bleu_recall_10)

            result = {
                'summary': 'bart',
                'bleu_precision_5': bleu_precision_5,
                'bleu_precision_10': bleu_precision_10,
                'bleu_recall_5': bleu_recall_5,
                'bleu_recall_10': bleu_recall_10,
                'bleu_f_5': bleu_f_5,
                'bleu_f_10': bleu_f_10
            }
        results.append(result)

    results = pandas.DataFrame.from_dict(results)
    g = results.groupby('summary')
    mean = g.mean()
    mean_dict = mean.T.to_dict()
    std = g.std().replace(numpy.nan, 0.0)
    std_dict = std.T.to_dict()

    if args.use_original_test_data:
        print(f'model & BLEU \\\\')
        for summary in sorted(mean_dict.keys()):
            print(f'{summary} & '
                  f'{mean_dict[summary]["bleu"]:.2f} $\\pm$ {std_dict[summary]["bleu"]:.2f} \\\\')
    else:
        print(f'model & P@5 & R@5 & F@5 & P@10 & R@10 & F@10 \\\\')
        for summary in sorted(mean_dict.keys()):
            print(
                f'{summary} & '
                f'{mean_dict[summary]["bleu_precision_5"]:.2f} $\\pm$ {std_dict[summary]["bleu_precision_5"]:.2f} & '
                f'{mean_dict[summary]["bleu_recall_5"]:.2f} $\\pm$ {std_dict[summary]["bleu_recall_5"]:.2f} & '
                f'{mean_dict[summary]["bleu_f_5"]:.2f} $\\pm$ {std_dict[summary]["bleu_f_5"]:.2f} & '
                f'{mean_dict[summary]["bleu_precision_10"]:.2f} $\\pm$ {std_dict[summary]["bleu_precision_10"]:.2f} & '
                f'{mean_dict[summary]["bleu_recall_10"]:.2f} $\\pm$ {std_dict[summary]["bleu_recall_10"]:.2f} & '
                f'{mean_dict[summary]["bleu_f_10"]:.2f} $\\pm$ {std_dict[summary]["bleu_f_10"]:.2f} \\\\')


if __name__ == '__main__':
    main()
