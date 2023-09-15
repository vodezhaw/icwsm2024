
import json
import gzip
from pathlib import Path

from icwsm2024.datasets.classifier import TrainDataset, TestDataset
from icwsm2024.classifiers.huggingface import finetune


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", dest="train", type=Path, required=True)
    parser.add_argument(
        "--test", dest="test", type=Path, required=True)
    parser.add_argument(
        "--scores", dest="scores", type=Path, required=True)
    parser.add_argument(
        "--model-save", dest="model_out", type=Path, required=True)
    parser.add_argument(
        "--hf-model", dest="hf_model", type=str, required=True)
    parser.add_argument(
        "--short-name", dest="short_name", type=str, required=True)
    parser.add_argument(
        "--dev", dest="dev", type=bool, required=False, default=False, action="store_true")
    args = parser.parse_args()

    main(
        train_path=args.train,
        test_path=args.test,
        scores_path=args.scores,
        model_out=args.model_out,
        hf_model=args.hf_model,
        model_short=args.short_name,
        dev_mode=args.dev,
    )
