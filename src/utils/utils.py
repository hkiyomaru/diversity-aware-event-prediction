from typing import List, Tuple, Dict, Optional, Union
import json
import os
import sys
from time import gmtime
from time import strftime

import chainer
import numpy
import gensim

from utils.constants import UNK


def count_lines(path: str) -> int:
    with open(path) as f:
        return sum([1 for _ in f])


def load_vocabulary(path: str) -> Dict[str, int]:
    with open(path) as f:
        # +2 for UNK and EOS
        word_ids = {line.strip(): i + 2 for i, line in enumerate(f)}
    word_ids['<UNK>'] = 0
    word_ids['<EOS>'] = 1
    return word_ids


def load_data(vocabulary: Dict[str, int],
              path: str,
              debug: bool = False,
              unique: bool = False,
              needs_raw_text: bool = False
              ) -> Union[List[numpy.ndarray], Tuple[List[numpy.ndarray], List[str]]]:
    data, raw = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            words = line.strip().split()
            array = numpy.array([vocabulary.get(w, UNK) for w in words], numpy.int32)
            if unique and _is_in(array, data):
                continue
            data.append(array)
            raw.append(line.strip())
            if debug and i == 10:
                break
    if needs_raw_text:
        return data, raw
    else:
        return data


def _is_in(array: numpy.ndarray, arrays: List[numpy.ndarray]) -> bool:
    for target in arrays:
        if len(array) != len(target):
            continue
        elif all(array == target):
            return True
    return False


def load_embedding(path: str, vocab: Dict[str, int]) -> numpy.ndarray:
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True, unicode_errors='ignore')
    embedding = numpy.zeros((len(vocab), model.vector_size), 'f')
    unk_indexes = []
    for k, v in vocab.items():
        if k in model.vocab:
            embedding[v] = model.word_vec(k)
        else:
            unk_indexes.append(v)
    if len(unk_indexes) > 0:
        unk_vector = numpy.sum(embedding, axis=0) / (len(vocab) - len(unk_indexes))
        embedding[unk_indexes] = unk_vector
    return embedding


def calculate_unknown_ratio(data: List[numpy.ndarray]) -> float:
    unknown = sum((s == UNK).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total


def convert(batch: List[Tuple[numpy.ndarray, numpy.ndarray]], device: Optional[int] = None) \
        -> Dict[str, List[numpy.ndarray]]:

    def to_device_batch(batch: List[numpy.ndarray]) -> List[numpy.ndarray]:
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = chainer.backends.cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x) for x in batch[:-1]], dtype=numpy.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = chainer.backends.cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return {'xs': to_device_batch([x for x, _ in batch]),
            'ys': to_device_batch([y for _, y in batch])}


def convert_line(vocabulary: Dict[str, int], line: str) -> numpy.ndarray:
    words = line.strip().split()
    array = numpy.array([vocabulary.get(w, UNK) for w in words], numpy.int32)
    return array


def make_output_dir(result_base: str, dirname: str) -> str:
    if dirname is None:
        dirname = strftime("%Y%m%d%H%M%S", gmtime())
    dirname = os.path.join(result_base, dirname)
    os.makedirs(dirname, exist_ok=True)
    return dirname


def save_config(output_dir: str, config) -> None:
    # write the execution command
    with open(os.path.join(output_dir, 'run.txt'), 'w') as f:
        f.write('python %s' % ' '.join(sys.argv))

    # write the parameters as a config file
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
