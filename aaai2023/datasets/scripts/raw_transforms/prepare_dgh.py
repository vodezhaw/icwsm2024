
import csv
from pathlib import Path

from aaai2023.datasets.classifier import Sample, TrainDataset, TestDataset


def prepare(
    raw: Path,
):

    with (raw / "dynamically_generated_hate" / "dynamically_generated_hate.csv").open('r') as fin:
        reader = csv.DictReader(
            fin,
            delimiter=",",
            quoting=csv.QUOTE_MINIMAL,
        )
        raw = list(reader)

    train_samples = [
        Sample(
            id=f"dgh-{r['acl.id']}",
            text=r['text'],
            label=r['label'] == 'hate',
        )
        for r in raw
        if r['split'] == 'train'
    ]
    dev_samples = [
        Sample(
            id=f"dgh-{r['acl.id']}",
            text=r['text'],
            label=r['label'] == 'hate',
        )
        for r in raw
        if r['split'] == 'dev'
    ]
    test_samples = [
        Sample(
            id=f"dgh-{r['acl.id']}",
            text=r['text'],
            label=r['label'] == 'hate',
        )
        for r in raw
        if r['split'] == 'test'
    ]

    train_data = TrainDataset(
        name="dynamically-generated-hate",
        train_samples=train_samples,
        dev_samples=dev_samples,
    )
    test_data = TestDataset(
        name="dynamically-generated-hate",
        test_samples=test_samples,
    )

    return train_data, test_data
