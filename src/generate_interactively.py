"""Evaluate seq2seq model iteratively."""
from typing import Dict
import argparse
import datetime
import os
import sys

import numpy
import chainer

from collections import defaultdict

from train import get_instance
import model as module_arch
from utils.constants import CHECK_POINT, UNK
from utils.utils import load_vocabulary
from test import load_configs
from show_result import check_result


def analyze(inp: str, lang: str) -> str:
    if lang == 'ja':
        raise NotImplementedError
    elif lang == 'en':
        return _analyze_en(inp)
    else:
        print(f'unknown language: {lang}')
        sys.exit(1)


def _analyze_en(inp: str) -> str:
    return inp


def convert_data(vocabulary: Dict[str, int], line: str) -> numpy.ndarray:
    words = line.strip().split()
    array = numpy.array([vocabulary.get(w, UNK) for w in words], numpy.int32)
    return array


def show_help() -> None:
    print('DESCRIPTION')
    print('    c, config\n'
          '        show configuration file')
    print('    h, help\n'
          '        show help')
    print('    q, quit\n'
          '        quit')
    print('    otherwise\n'
          '        generate following events')


def main():
    parser = argparse.ArgumentParser(description='Generate translation interactively')
    parser.add_argument('RESULTS', help='path pattern to result directories')
    parser.add_argument('--model-selection', default='loss', choices=['loss', 'bleu'],
                        help='model selection')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--show-frequency', action='store_true',
                        help='show each frequency of results from 100 times samplings')
    parser.add_argument('--num-sampling', default=10, type=int,
                        help='number of sampling sequences')
    args = parser.parse_args()

    # load configuration files with sorting metrics
    configs = load_configs(args.RESULTS, args.model_selection)

    # load dataset
    config, _, _ = configs[sorted(configs.keys())[0]]
    print(f'[{datetime.datetime.now()}] Loading dataset... (this may take several minutes)')
    word_ids = load_vocabulary(config['data']['vocabulary'])

    source_words = {i: w for w, i in word_ids.items()}
    target_words = {i: w for w, i in word_ids.items()}

    print(f'[{datetime.datetime.now()}] Dataset loaded.')

    id2config = {i: key for i, key in enumerate(sorted(configs.keys()))}
    while True:
        print('Select model architecture')
        for i, key in id2config.items():
            print(f' {i:2d}: {key}')
        inp = input('> ')
        if inp.isdigit() is False or int(inp) not in id2config:
            continue

        print(f'[{datetime.datetime.now()}] Resuming model ({inp}) {id2config[int(inp)]}...')
        config, result_dir, _ = configs[id2config[int(inp)]]
        model = get_instance(module_arch, 'arch', config, source_initialW=None, target_initialW=None)

        model_path = os.path.join(result_dir, CHECK_POINT[args.model_selection])
        chainer.serializers.load_npz(model_path, model)

        print(f'[{datetime.datetime.now()}] Model loaded.')

        if args.gpu >= 0:
            chainer.backends.cuda.get_device_from_id(args.gpu).use()
            model.to_gpu(args.gpu)

        show_help()
        while True:
            inp = input('> ')
            if inp == '':
                continue
            elif inp == 'h' or inp == 'help':
                show_help()
                continue
            elif inp == 'c' or inp == 'config':
                check_result(result_dir, args.model_selection)
                continue
            elif inp == 'q' or inp == 'quit':
                break

            source = convert_data(word_ids, analyze(inp, config['data']['lang']))
            print('[Parsed input]', ' '.join([source_words[x] for x in source]))

            if config['arch']['type'] == 'CVAESeq2seq':
                _, var = model.translate([model.xp.array(source)], needs_hv=True, sampling=True)
                print(f'[Logarithm of hyper-volume] {var[0]}')

            generate_dict = defaultdict(int)
            for k in range(args.num_sampling):
                result = model.translate([model.xp.array(source)], sampling=True)[0]
                result = ' '.join([target_words[y] for y in result])

                if args.show_frequency:
                    generate_dict[result] += 1
                else:
                    print(f'[Generated #{k}] {result}')

            if args.show_frequency:
                for event, count in sorted(generate_dict.items(), key=lambda x: -x[1]):
                    print(f'{event} ({count})')


if __name__ == '__main__':
    main()
