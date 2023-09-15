
import csv
import io
import zipfile
from pathlib import Path

from icwsm2024.datasets.classifier import Sample, TrainDataset, TestDataset


def prepare(
    raw: Path
):

    splits = {}
    with zipfile.ZipFile(raw / "wassa" / "wassa.zip", 'r') as zipf:
        for split in ["train", "test"]:
            with io.TextIOWrapper(zipf.open(f"{split}.tsv", 'r'), encoding='utf-8') as fin:
                reader = csv.DictReader(
                    fin,
                    delimiter="\t",
                    quoting=csv.QUOTE_MINIMAL,
                )
                splits[split] = list(reader)

    for _, ss in splits.items():
        assert all(s['HOF'] in {'Hateful', "Non-Hateful"} for s in ss)

    train_samples = [
        Sample(
            id=f"wassa-{ix}",
            text=d['text'],
            label=d['HOF'] == 'Hateful'
        )
        for ix, d in enumerate(splits['train'])
    ]
    test_samples = [
        Sample(
            id=f"wassa-{len(train_samples) + ix}",
            text=d['text'],
            label=d['HOF'] == "Hateful",
        )
        for ix, d in enumerate(splits['test'])
    ]

    train_data = TrainDataset(
        name="wassa",
        train_samples=train_samples,
        dev_samples=None,
    )
    test_data = TestDataset(
        name="wassa",
        test_samples=test_samples,
    )

    return train_data, test_data
