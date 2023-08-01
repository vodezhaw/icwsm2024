
import json
import gzip
from pathlib import Path

from aaai2023.datasets.classifier import TrainDataset, TestDataset
from aaai2023.classifiers.huggingface import finetune


def main(
    train_path: Path,
    test_path: Path,
    scores_path: Path,
    model_out: Path,
    model_short: str,
    hf_model: str,
    dev_mode: bool = False,
):
    train_datasets = []
    for f in train_path.glob("*.json.gz"):
        with gzip.open(f, "rt") as fin:
            train_datasets.append(TrainDataset.from_json(json.load(fin)))

    test_datasets = []
    for f in test_path.glob("*.json.gz"):
        with gzip.open(f, "rt") as fin:
            test_datasets.append(TestDataset.from_json(json.load(fin)))

    if dev_mode:
        train_datasets = train_datasets[:1]
        test_datasets = test_datasets[:1]

    for train_data in train_datasets:
        print(f"training {model_short} on {train_data.name}")
        finetune(
            model_short_name=model_short,
            base_model=hf_model,
            train_data=train_data,
            test_data=test_datasets,
            scores_dir=scores_path,
            model_dir=model_out,
            dev_mode=dev_mode,
        )
