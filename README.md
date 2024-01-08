
# ICWSM 2024

This repository was developed with [Poetry](https://python-poetry.org/) for package management.
The dependencies are listed in `pyproject.toml`. If you use `poetry` you might
need to prepend `poetry run` to the `python` commands below.

Developed for `python 3.11.X`

## Prepare Data

You can find information on how to obtain the datasets in this work in the
`data/raw/` folder. There is one sub-folder for each dataset. Where possible
we included a `download.sh` script that will download the raw data. Otherwise
we left a note about how to access the data.

To prepare the raw data, run the following script:

```shell
python -m icwsm2024.datasets.scripts.prepare_raw --raw ./data/raw/ --train ./data/train/ --test ./data/test/
```

## Train Classifiers and get Scores

### Electra & RoBERTa

Training Electra:
```shell
python -m icwsm2024.classifiers.scripts.finetune_full --train ./data/train --test ./data/test --scores ./data/scores --model-save /some/path/ --hf-model google/electra-base-discriminator --short-name electra
```

Training Twitter-RoBERTA:
```shell
python -m icwsm2024.classifiers.scripts.finetune_full --train ./data/train --test ./data/test --scores ./data/scores --model-save /some/path/ --hf-model cardiffnlp/twitter-roberta-base --short-name cardiffnlp
```

### TF-IDF SVM

Training TF-IDF SVM models:
```shell
python -m icwsm2024.classifiers.scripts.train_tfidf --train ./data/train --test ./data/test --scores ./data/scores
```

### Perspective API

To use [Perspective API](https://www.perspectiveapi.com/) you have to request an
API key.

We use the follwoing script to query the API:

```shell
python -m icwsm2024.classifiers.scripts.run_perspective --test ./data/test --scores ./data/scores --api-key YOUR-API-KEY
```

We cache the API results in a `perspective-cache.shelve` file in the current directory.

## Subsample Prevalences

We precompute the sub-sampled versions of the various datasets:

```shell
python -m icwsm.datasets.scripts.subsample_hate_data --test ./data/test --scores ./data/scores
```

## Run Experiments

We run our experiment configurations in parallel using *gnu parallel* [0].

```shell
./bash/experiments.sh
```

[0]
```
@book{tange_ole_2018_1146014,
  author       = {Tange, Ole},
  title        = {GNU Parallel 2018},
  publisher    = {Ole Tange},
  year         = 2018,
  month        = apr,
  doi          = {10.5281/zenodo.1146014},
  url          = {https://doi.org/10.5281/zenodo.1146014}
}
```

## Generate Plots

Generate all plots used in the paper based on the experiment runs:

```shell
python -m icwsm2024.paper.plot_all
```
