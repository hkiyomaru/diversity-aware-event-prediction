"""Evaluate seq2seq model."""
import argparse
import datetime
import os
import progressbar
import collections

import numpy
import chainer
from chainer.backends import cuda

from train import get_instance
import model as module_arch
from utils.constants import CHECK_POINT
from utils.utils import load_vocabulary, load_data
from test import load_configs
from show_result import check_result


def make_surface(words, ts) -> str:
    return ' '.join(words[w] for w in ts)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate translation')
    parser.add_argument('RESULTS', help='path pattern to result directories')
    parser.add_argument('--model-selection', default='loss', choices=['loss', 'bleu', 'latest'], help='model selection')
    parser.add_argument('--input', '-i', default='', help='path to source sentence file (default: test data)')
    parser.add_argument('--output', '-o', default='', help='path to resultant file directory (default: model path)')
    parser.add_argument('--suffix', default='generation_result.tsv', help='resultant file name suffix')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--sampling', action='store_true', help='whether to sample sequence')
    parser.add_argument('--num-sampling', default=100, type=int, help='number of sampling')
    parser.add_argument('--num-example', default=5, type=int, help='number of sampling')
    parser.add_argument('--debug', action='store_true', help='whether to use small dataset')
    args = parser.parse_args()

    # load configuration files with sorting metrics
    configs = load_configs(args.RESULTS, args.model_selection)

    # load dataset
    config, _, _ = configs[sorted(configs.keys())[0]]
    print(f'[{datetime.datetime.now()}] Loading dataset... (this may take several minutes)')
    word_ids = load_vocabulary(config['data']['vocabulary'])
    source_words = {i: w for w, i in word_ids.items()}
    target_words = {i: w for w, i in word_ids.items()}

    test_source_path = args.input if args.input else config['data']['test_source']
    test_source, test_source_raw = load_data(word_ids, test_source_path, debug=args.debug, needs_raw_text=True)

    print(f'[{datetime.datetime.now()}] Dataset loaded.')

    for summary, (config, result_dir, _) in configs.items():
        print(f'[{datetime.datetime.now()}] Resuming model {summary}...')
        model = get_instance(
            module_arch,
            'arch',
            config,
            source_initialW=None,
            target_initialW=None
        )

        model_path = os.path.join(result_dir, CHECK_POINT[args.model_selection])
        chainer.serializers.load_npz(model_path, model)

        print(f'[{datetime.datetime.now()}] Model loaded')
        check_result(result_dir, args.model_selection)

        if args.gpu >= 0:
            chainer.backends.cuda.get_device_from_id(args.gpu).use()
            model.to_gpu(args.gpu)

        print(f'[{datetime.datetime.now()}] Start translation')
        bar = progressbar.ProgressBar()
        sources, targets = [], []
        if args.sampling:
            n_loop = len(test_source)
            for i in bar(range(len(test_source)), max_value=n_loop):
                source, source_row = test_source[i], test_source_raw[i]
                source_batch = [source] * args.num_sampling

                if args.gpu >= 0:
                    source_batch = cuda.to_gpu(source_batch)

                ys = [y.tolist() for y in model.translate(source_batch, sampling=True)]
                results = [make_surface(target_words, y) for y in ys]

                generate_dict = collections.defaultdict(int)
                for r in results:
                    generate_dict[r] += 1

                generateds = numpy.array(list(generate_dict.keys()))
                frequencies = numpy.array(list(generate_dict.values()))
                probabilities = frequencies / frequencies.sum()

                if len(generateds) < args.num_example:
                    print(f'WARNING: {i + 1}-th sentence gives not enough number of examples '
                          f'({len(generateds)} < {args.num_example})')
                    generateds = generateds.tolist()
                else:
                    generateds = numpy.random.choice(generateds, size=args.num_example, replace=False, p=probabilities)

                targets.extend(generateds)
                sources.extend([source_row] * len(generateds))
        else:
            n_loop = (len(test_source) // int(config['param']['batchsize'])) + 1
            for i in bar(range(0, len(test_source), int(config['param']['batchsize'])), max_value=n_loop):
                source = test_source[i:i + int(config['param']['batchsize'])]
                source_raw = test_source_raw[i:i + int(config['param']['batchsize'])]
                source_batch = source

                if args.gpu >= 0:
                    source_batch = cuda.to_gpu(source_batch)

                ys = model.translate(source_batch, sampling=False)
                results = [make_surface(target_words, y) for y in ys]

                targets.extend(results)
                sources.extend(source_raw)

        output_dir = args.output if args.output else result_dir
        result_filename = f'{summary}-{args.suffix}'
        result_filepath = os.path.join(output_dir, result_filename)

        last_source = ''
        with open(result_filepath, 'w') as f:
            for source, target in zip(sources, targets):
                if last_source != '' and last_source != source:
                    f.write('\n')
                f.write(f'{source}\t{target}\n')
                last_source = source
        print('Done')
        print(result_filepath)


if __name__ == '__main__':
    main()
