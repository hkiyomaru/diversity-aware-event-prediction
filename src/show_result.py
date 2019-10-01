import argparse
import json
import os

import numpy


METRICS_SORT_FUNC = {
    'loss': ('validation/main/loss', numpy.nanargmin),
    'bleu': ('validation/main/bleu', numpy.nanargmax),
    'latest': ('iteration', numpy.nanargmax)
}


def parse_result(path: str, model_selection: str) -> dict:
    with open(os.path.join(path, 'run.txt')) as f:
        command = f.read()

    # load config
    with open(os.path.join(path, 'config.json')) as f:
        config = json.load(f)

    # load log
    with open(os.path.join(path, 'log')) as f:
        log = json.load(f)

    indicator, sort_func = METRICS_SORT_FUNC[model_selection]
    best_model_idx = sort_func([item[indicator] if 'validation/main/loss' in item else numpy.nan for item in log])
    best_model_epoch = log[best_model_idx]['epoch']
    best_model_iter = log[best_model_idx]['iteration']
    best_model_score = log[best_model_idx][indicator]
    best_model_loss = log[best_model_idx]['validation/main/loss']
    best_model_bleu = log[best_model_idx]['validation/main/bleu']
    best_model_distinct1 = log[best_model_idx]['validation/main/distinct1']
    best_model_distinct2 = log[best_model_idx]['validation/main/distinct2']
    max_epoch = max([item['epoch'] for item in log])
    max_iter = max([item['iteration'] for item in log])
    result_info = {
        'path': path,
        'command': command,
        'dataset': config["data"]["base"],
        'lang': config['data']['lang'],
        'model': config['arch']['type'],
        'model_params': config['arch']['args'],
        'updater': config['updater']['type'],
        'updater_params': config['updater']['args'],
        'metrics': model_selection,
        'score': best_model_score,
        'loss': best_model_loss,
        'bleu': best_model_bleu,
        'distinct-1': best_model_distinct1,
        'distinct-2': best_model_distinct2,
        'epoch': best_model_epoch,
        'iter': best_model_iter,
        'max_epoch': max_epoch,
        'max_iter': max_iter}
    return result_info


def check_result(path: str, model_selection: str = 'loss', indent=2) -> None:
    result_info = parse_result(path, model_selection)
    _indent = ' ' * indent

    def params_to_str(params):
        if params:
            return "\n".join(_indent + f"{k}: {v}" for k, v in params.items())
        else:
            return _indent + "default"

    print(f"[Result directory]")
    print(_indent + f"{result_info['path']}")
    print(f"[Dataset directory]")
    print(_indent + f"{result_info['dataset']}")
    print(f"[Dataset language]")
    print(_indent + f"{result_info['lang']}")
    print(f"[Execution command]")
    print(_indent + f"{result_info['command']}")
    print(f"[Model]")
    print(_indent + f"{result_info['model']}")
    print(f"[Model parameters]")
    print(f"{params_to_str(result_info['model_params'])}")
    print(f"[Updater]")
    print(_indent + f"{result_info['updater']}")
    print(f"[Updater parameters]")
    print(f"{params_to_str(result_info['updater_params'])}")
    print(f"[Best model statistics]")
    print(_indent + f"epoch: {result_info['epoch']} (max: {result_info['max_epoch']})")
    print(_indent + f"iteration: {result_info['iter']} (max: {result_info['max_iter']})")
    print(_indent + f"score: {result_info['score']:.4f} ({result_info['metrics']})")
    print(_indent + f"loss: {result_info['loss']:.4f}")
    print(_indent + f"BLEU: {result_info['bleu'] * 100:.2f}")
    print(_indent + f"distinct-1: {result_info['distinct-1'] * 100:.2f}")
    print(_indent + f"distinct-2: {result_info['distinct-2'] * 100:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('RESULT', help='path to result directory')
    args = parser.parse_args()
    check_result(args.RESULT)
