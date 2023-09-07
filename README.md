
# ICWSM 2024

## Prepare Data

You can find information on how to obtain the datasets in this work in the
`data/raw/` folder. There is one sub-folder for each dataset. Where possible
we included a `download.sh` script that will download the raw data. Otherwise
we left a note about how to access the data.

To prepare the raw data, run the following script:

```shell
python -m aaai2023.datasets.scripts.prepare_raw --raw ./data/raw/ --train ./data/train/ --test ./data/test/
```

## Train Classifiers and get Scores

### Electra & RoBERTa

### TF-IDF SVM

### Perspective API

## Subsample Prevalences

## Run Experiments

## Generate Plots