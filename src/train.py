"""Train seq2seq model."""
import argparse
import json

import numpy
import chainer
from chainer.backends import cuda
from chainer import training
from chainer.training import extensions

from utils import load_vocabulary, load_data, load_embedding
from utils import convert
from utils import make_output_dir
from utils import save_config

import model as module_arch
import updater as module_updater

from utils.metrics import CalculateMetrics
from utils.metrics import calculate_bleu, calculate_distinct_1, calculate_distinct_2


def get_instance(module, name, config, *args, **kwargs):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'], **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description='training a seq2seq model')
    parser.add_argument('CONFIG', help='path to config file')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    with open(args.CONFIG, 'r') as f:
        config = json.load(f)

    log_dir = make_output_dir(config['param']['result'], config['param']['output'])
    save_config(log_dir, config)

    word_ids = load_vocabulary(config['data']['vocabulary'])
    target_words = {i: w for w, i in word_ids.items()}
    source_words = {i: w for w, i in word_ids.items()}

    train_source = load_data(word_ids, config['data']['train_source'])
    train_target = load_data(word_ids, config['data']['train_target'])
    assert len(train_source) == len(train_target)
    train_data = [(s, t) for s, t in zip(train_source, train_target) if 0 < len(s) and 0 < len(t)]

    train_iter = chainer.iterators.MultithreadIterator(train_data, config['param']['batchsize'], n_threads=4)

    initialW = load_embedding(config['data']['source_word_embedding'], word_ids)
    model = get_instance(
        module_arch,
        "arch",
        config,
        source_initialW=initialW,
        target_initialW=initialW
    )
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    optimizer = chainer.optimizers.Adam(alpha=config['param']['learning_rate'])
    optimizer.setup(model)
    updater = get_instance(
        module_updater,
        "updater",
        config,
        train_iter,
        optimizer,
        converter=convert,
        device=args.gpu
    )

    trainer = training.Trainer(
        updater,
        chainer.training.triggers.EarlyStoppingTrigger(
            patients=config['param']['patients'],
            monitor='validation/main/loss',
            max_trigger=(config['param']['epoch'], 'epoch')
        ),
        out=log_dir
    )
    trainer.extend(extensions.LogReport(trigger=(config['param']['log_interval'], 'epoch')))
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'iteration',
             'main/loss', 'main/rec_loss', 'main/reg_loss', 'main/perp',
             'validation/main/loss', 'validation/main/rec_loss', 'validation/main/reg_loss', 'validation/main/perp',
             'validation/main/bleu', 'validation/main/distinct1', 'validation/main/distinct2',
             'elapsed_time']
        ),
        trigger=(config['param']['log_interval'], 'epoch')
    )
    trainer.extend(
        extensions.snapshot_object(model, 'seq2seq.best_bleu.npz'),
        trigger=chainer.training.triggers.MaxValueTrigger(
            key='validation/main/bleu',
            trigger=(config['param']['validation_interval'], 'epoch')
        )
    )
    trainer.extend(
        extensions.snapshot_object(model, 'seq2seq.best_loss.npz'),
        trigger=chainer.training.triggers.MinValueTrigger(
            key='validation/main/loss',
            trigger=(config['param']['validation_interval'], 'epoch')
        )
    )
    trainer.extend(
        extensions.snapshot_object(model, 'seq2seq.latest.npz'),
        trigger=(config['param']['validation_interval'], 'epoch')
    )

    test_source = load_data(word_ids, config['data']['valid_source'])
    test_target = load_data(word_ids, config['data']['valid_target'])
    assert len(test_source) == len(test_target)
    test_data = [(s, t) for s, t in zip(test_source, test_target) if 0 < len(s) and 0 < len(t)]

    trainer.extend(
        CalculateMetrics(
            model,
            test_data,
            ('validation/main/bleu', 'validation/main/distinct1', 'validation/main/distinct2'),
            (calculate_bleu, calculate_distinct_1, calculate_distinct_2),
            device=args.gpu
        ),
        trigger=(config['param']['validation_interval'], 'epoch')
    )
    trainer.extend(
        extensions.Evaluator(
            chainer.iterators.SerialIterator(test_data, config['param']['batchsize'], repeat=False),
            model,
            device=args.gpu,
            converter=convert
        ),
        trigger=(config['param']['validation_interval'], 'epoch')
    )

    @chainer.training.make_extension()
    def translate(_):
        source, target = test_data[numpy.random.choice(len(test_data))]
        result = model.translate([model.xp.array(source)], sampling=False)[0]
        source_sentence = ' '.join([source_words[x] for x in source])
        target_sentence = ' '.join([target_words[y] for y in target])
        result_sentence = ' '.join([target_words[y] for y in result])
        print(f'# source : {source_sentence}')
        print(f'# result : {result_sentence}')
        print(f'# expect : {target_sentence}')

    trainer.extend(
        translate,
        trigger=(config['param']['validation_interval'], 'epoch')
    )

    trainer.run()


if __name__ == '__main__':
    main()
