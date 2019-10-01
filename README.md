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

The configuration used in the experiments are in `./config`.
To start training, run `./src/train.py` with a configuration.
The results will be written into `./result`

```
$ pipenv run python src/train.py "config/descript/descript-Seq2seq-batchsize.64-epoch.100-lr.0.001-n_layers.2-n_units.300"
```

### Test

To evaluate trained models, run `./src/test.py` with a glob pattern for result directories.

```
$ pipenv run python src/test.py "./result/descript/*"  # specify a pattern to glob result directories
```

To evaluate trained models leaned with different random seeds, run `./src/test.py` with multiple glob patterns for result directories.

```
$ pipenv run python src/test.py "./result/descript-1/*" "./result/descript-2/*" "config/descript-3/*"
```

### Generation

To generate next events with a trained model, run `./src/generate_interactively.py` with a glob pattern for result directories.

```
$ pipenv run python src/generate_interactively.py "./result/descript/*"
```

## License

- code: MIT License
- data: GNU General Public License, version 2

## Reference

```
Hirokazu Kiyomaru, Kazumasa Omura, Yugo Murawaki, Daisuke Kawahara and Sadao Kurohashi:
Diversity-aware Event Prediction based on a Conditional Variational Autoencoder with Reconstruction,
Proceedings of COIN: COmmonsense INference in Natural Language Processing, Hong Kong, 2019.
```
