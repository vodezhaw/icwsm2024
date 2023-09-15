
import csv
import html
from pathlib import Path

from sklearn.model_selection import train_test_split

from icwsm2024.datasets.classifier import TrainDataset, TestDataset, Sample


def read_raw(
    raw: Path,
):
    with (raw / "davidson" / "davidson.csv").open('r') as fin:
        reader = csv.DictReader(
            fin,
            delimiter=",",
            quoting=csv.QUOTE_MINIMAL,
        )
        raw = list(reader)

    assert all(r['class'] in {'0', '1', '2'} for r in raw)

    return train_test_split(
        raw,
        test_size=.1,
        random_state=0xdeadbeef,
        stratify=[r['class'] for r in raw],
    )


def prepare_hate_only(
    raw: Path,
):
    train_data, test_data = read_raw(raw)
    train_hate_only = TrainDataset(
        name="davidson-hate-only",
        train_samples=[
            Sample(
                id=f"davidson-hate-only-{r['']}",
                text=html.unescape(r['tweet']),
                label=r['class'] == '0',
            )
            for r in train_data
        ],
        dev_samples=None,
    )
    test_hate_only = TestDataset(
        name="davidson-hate-only",
        test_samples=[
            Sample(
                id=f"davidson-hate-only-{r['']}",
                text=html.unescape(r['tweet']),
                label=r['class'] == '0',
            )
            for r in test_data
        ],
    )
    return train_hate_only, test_hate_only


def prepare_hate_off(
    raw: Path,
):
    train_data, test_data = read_raw(raw)
    train_hate_off = TrainDataset(
        name="davidson-hate-off",
        train_samples=[
            Sample(
                id=f"davidson-hate-off-{r['']}",
                text=html.unescape(r['tweet']),
                label=r['class'] != '2',
            )
            for r in train_data
        ],
        dev_samples=None,
    )
    test_hate_off = TestDataset(
        name="davidson-hate-off",
        test_samples=[
            Sample(
                id=f"davidson-hate-off-{r['']}",
                text=html.unescape(r['tweet']),
                label=r['class'] != '2',
            )
            for r in test_data
        ],
    )
    return train_hate_off, test_hate_off
