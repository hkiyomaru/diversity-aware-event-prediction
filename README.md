# Diversity-aware Event Prediction based on a Conditional Variational Autoencoder with Reconstruction

## Development Environment

- Ubuntu 16.04
- Python 3.6.5
- chainer 5.4.0
- cupy 5.4.0
- gensim 3.8.0
- scipy 1.3.0
- nltk 3.4.5
- pandas 0.25.0
- progressbar2 3.42.0

## Getting Started

### Installation

```
$ pip install pipenv --user
$ pipenv install
```

### Downloading Dataset

```
$ ./download.sh  # the resultant data size will be 3.5GB
```

### Training

The configuration files used in the experiments are in `./config`.
To start training, run `./src/train.py` with a configuration file.
The results will be written into `./result`

```
$ pipenv run python src/train.py "config/descript/descript-Seq2seq-batchsize.64-epoch.100-lr.0.001-n_layers.2-n_units.300"
```

### Test

To evaluate trained models, run `./src/test.py` with a glob pattern to grep result directories.

```
$ pipenv run python src/test.py "./result/descript/*"  # specify a pattern to glob result directories
```

To evaluate trained models leaned with different random seeds, run `./src/test.py` with multiple glob patterns.

```
$ pipenv run python src/test.py "./result/descript-seed-0/*" "./result/descript-seed-1/*" "config/descript-seed-2/*"
```

### Generation

To generate next events using a trained model, run `./src/generate_interactively.py` with a glob pattern to grep result directories.

```
$ pipenv run python src/generate_interactively.py "./result/descript/*"
```

## License

- code: MIT License
- data: GNU General Public License, version 2

## Reference

```
Hirokazu Kiyomaru, Kazumasa Omura, Yugo Murawaki, Daisuke Kawahara and Sadao Kurohashi.
Diversity-aware Event Prediction based on a Conditional Variational Autoencoder with Reconstruction.
In Proceedings of the First Workshop on Commonsense Inference in Natural Language Processing (COIN), pp. 113-122, Hong Kong, November 2019.
```
