# Diversity-aware Event Prediction based on a Conditional Variational Autoencoder with Reconstruction

A reimplementation of Kiyomaru et al. (2019) using BART (Lewis+, 2019).

## Development Environment

- Ubuntu 20.04
- Python 3.9.7
- torch 1.10
- transformers 4.15.0

## Getting Started

### Installation

```
$ poetry install
```

### Downloading Dataset

```
$ ./download.sh  # the resultant data size will be 3.5GB
```

### Training

The configuration files used in the experiments are in `./config`.
To start training, run `./src/train.py` with a configuration file.
The results will be written into `./result`

TBW

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
@inproceedings{kiyomaru-etal-2019-diversity,
    title = "Diversity-aware Event Prediction based on a Conditional Variational Autoencoder with Reconstruction",
    author = "Kiyomaru, Hirokazu  and
      Omura, Kazumasa  and
      Murawaki, Yugo  and
      Kawahara, Daisuke  and
      Kurohashi, Sadao",
    booktitle = "Proceedings of the First Workshop on Commonsense Inference in Natural Language Processing",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-6014",
    doi = "10.18653/v1/D19-6014",
    pages = "113--122",
    abstract = "Typical event sequences are an important class of commonsense knowledge. Formalizing the task as the generation of a next event conditioned on a current event, previous work in event prediction employs sequence-to-sequence (seq2seq) models. However, what can happen after a given event is usually diverse, a fact that can hardly be captured by deterministic models. In this paper, we propose to incorporate a conditional variational autoencoder (CVAE) into seq2seq for its ability to represent diverse next events as a probabilistic distribution. We further extend the CVAE-based seq2seq with a reconstruction mechanism to prevent the model from concentrating on highly typical events. To facilitate fair and systematic evaluation of the diversity-aware models, we also extend existing evaluation datasets by tying each current event to multiple next events. Experiments show that the CVAE-based models drastically outperform deterministic models in terms of precision and that the reconstruction mechanism improves the recall of CVAE-based models without sacrificing precision.",
}
```
