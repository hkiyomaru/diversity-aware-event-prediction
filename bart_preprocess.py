import pathlib
import json

data_dir = pathlib.Path('data')

datasets = ('descript', 'wikihow')
splits = ('train', 'dev', 'test')


def load(path: pathlib.Path) -> list[str]:
    with path.open() as f:
        return [line.strip() for line in f]


for dataset in datasets:
    for split in splits:
        source_path = data_dir / dataset / f'{split}.source'
        target_path = data_dir / dataset / f'{split}.target'

        sources = load(source_path)
        targets = load(target_path)

        with (data_dir / dataset / f'{split}.json').open('wt') as f:
            for source, target in zip(sources, targets):
                f.write(json.dumps({'source': source, 'target': target}) + '\n')
