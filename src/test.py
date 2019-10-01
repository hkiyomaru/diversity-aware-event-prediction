"""Evaluate seq2seq model."""
from typing import List, Tuple
import argparse
import json
import os
import glob
import progressbar
import collections
import itertools

import numpy
import pandas
import chainer
from chainer.backends import cuda

from train import get_instance
import model as module_arch
from utils.constants import CHECK_POINT
from utils.metrics import get_bleu_precision_func
from utils.metrics import get_bleu_recall_func
from utils.utils import load_vocabulary
from utils.utils import load_data
from show_result import check_result
from show_result import METRICS_SORT_FUNC


seed = 42
numpy.random.seed(seed)
cuda.cupy.random.seed(seed)
os.environ['CHAINER_SEED'] = str(seed)
chainer.global_config.use_cudnn = 'never'


def load_configs(path_pattern: str, model_selection: str) -> dict:
    configs = {}

    parity = ''
    for path in glob.glob(path_pattern):
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
        _parity = config['data']['base']
        if parity == '':
            parity = _parity
        elif parity != _parity:
            raise ValueError(f'results should have learned on the same dataset, but {parity} != {_parity}')

    for path in glob.glob(path_pattern):
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)

        with open(os.path.join(path, 'log'), 'r') as f:
            log = json.load(f)

        # check best model over hyper-parameters
        summary = _model_architecture_summary(config)
        try:
            current_best = _best_metrics(log, model_selection)
        except ValueError:
            continue

        if summary in configs:
            _, _, best = configs[summary]
            if _needs_update(best, current_best, model_selection):
                configs[summary] = (config, path, current_best)
        else:
            configs[summary] = (config, path, current_best)

    return configs


def _model_architecture_summary(config: dict) -> str:
    if config['arch']['type'] == 'Seq2seq':
        summary = 'S2S'
    elif config['arch']['type'] == 'CVAESeq2seq':
        summary = 'CVAE'
    else:
        raise KeyError(f'unknown architecture type was passed:', config['arch']['type'])

    if config['arch']['args']['attention']:
        summary += '+att'
    if config['arch']['args']['reconstruction']:
        weight = config['updater']['args'].get('lambda_reconstruction', 1.0)
        summary += f'+rec ($\\lambda = {weight}$)'
    return summary


def _best_metrics(log: dict, model_selection: str) -> float:
    indicator, sort_func = METRICS_SORT_FUNC[model_selection]
    best_model_idx = sort_func([item[indicator] if 'validation/main/loss' in item else numpy.nan for item in log])
    return log[best_model_idx][indicator]


def _needs_update(best: float, current_best: float, model_selection: str) -> bool:
    if model_selection == 'loss':
        return best > current_best
    elif model_selection == 'bleu':
        return best < current_best
    else:
        raise KeyError(f'unknown model selection was passed: {model_selection}')


def aggregate_data(sources: List[numpy.ndarray],
                   targets: List[numpy.ndarray]
                   ) -> Tuple[List[numpy.ndarray], List[List[numpy.ndarray]]]:
    source2pairs = collections.defaultdict(list)
    for source, target in zip(sources, targets):
        source2pairs[tuple(source)].append(target)

    sources_, targets_ = [], []
    for s, ts in source2pairs.items():
        sources_.append(numpy.array(s))
        targets_.append(ts)
    return sources_, targets_


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate test results in a latex format')
    parser.add_argument('RESULTS', nargs='*', help='path pattern to result directories')
    parser.add_argument('--model-selection', default='loss', choices=['loss', 'bleu'], help='model selection')
    parser.add_argument('--num-example', default=15, type=int, help='number of sampling')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--debug', action='store_true', help='whether to use small dataset')
    parser.add_argument('--use-original-test-data', action='store_true', help='whether to use original test set')
    parser.add_argument('--batchsize', default=-1, type=int, help='the number of batchsize')
    args = parser.parse_args()

    # load configuration files with sorting metrics
    configs = [load_configs(path_pattern, args.model_selection) for path_pattern in args.RESULTS]

    # load dataset
    configs_ = configs[0]  # extract first configs
    config, result_dir, _ = configs_[sorted(configs_.keys())[0]]
    word_ids = load_vocabulary(config['data']['vocabulary'])

    if args.use_original_test_data:
        test_source_path = config['data']['test_source']
        test_target_path = config['data']['test_target']
    else:
        test_source_path = os.path.join(config['data']['base'], 'mturk.source')
        test_target_path = os.path.join(config['data']['base'], 'mturk.target')
    test_source = load_data(word_ids, test_source_path, debug=args.debug)
    test_target = load_data(word_ids, test_target_path, debug=args.debug)
    assert len(test_source) == len(test_target)

    if args.use_original_test_data:
        test_source, test_target = test_source, [[t] for t in test_target]
    else:
        test_source, test_target = aggregate_data(test_source, test_target)

    results = []
    for configs_ in configs:
        for summary, (config, result_dir, _) in configs_.items():
            check_result(result_dir, args.model_selection)
            model = get_instance(module_arch, "arch", config, source_initialW=None, target_initialW=None)

            model_path = os.path.join(result_dir, CHECK_POINT[args.model_selection])
            chainer.serializers.load_npz(model_path, model)

            if args.gpu >= 0:
                chainer.backends.cuda.get_device_from_id(args.gpu).use()
                model.to_gpu(args.gpu)

            batch_size = args.batchsize if args.batchsize > 0 else config['param']['batchsize']
            n_repeat = 1 if args.use_original_test_data else args.num_example
            n_source_in_batch = batch_size if args.use_original_test_data else batch_size // args.num_example
            n_loop = len(test_source) // n_source_in_batch
            if len(test_source) % n_source_in_batch > 0:
                n_loop += 1  # round up to evaluate all test data

            references, hypotheses = [], []
            bar = progressbar.ProgressBar()
            for i in bar(range(n_loop), max_value=n_loop):
                sources = test_source[i * n_source_in_batch:(i + 1) * n_source_in_batch]
                targets = test_target[i * n_source_in_batch:(i + 1) * n_source_in_batch]

                repeated_source_batch = [itertools.repeat(s, n_repeat) for s in sources]
                source_batch = [source for repeated_source in repeated_source_batch for source in repeated_source]

                if args.gpu >= 0:
                    source_batch = cuda.to_gpu(source_batch)

                sampling = False if args.use_original_test_data else True

                ys = model.translate(source_batch, sampling=sampling)

                # split results so that each element has a size of "n_repeat"
                n_source_in_this_batch = len(sources)
                ys = [ys[i * n_repeat:(i + 1) * n_repeat] for i in range(n_source_in_this_batch)]

                hypotheses.extend([[h.tolist() for h in hs] for hs in ys])
                references.extend([[r.tolist() for r in rs] for rs in targets])

            if args.use_original_test_data:
                bleu = get_bleu_precision_func(references, hypotheses)(at=1, offset=0) * 100
                result = {
                    'summary': summary,
                    'bleu': bleu
                }
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
                    'summary': summary,
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
