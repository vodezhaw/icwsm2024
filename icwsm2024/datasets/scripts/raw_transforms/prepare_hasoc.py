
from pathlib import Path
import csv
import zipfile
import io

from icwsm2024.datasets.classifier import Sample, TrainDataset, TestDataset


def prepare(
    raw: Path,
):
    with zipfile.ZipFile(raw / "hasoc2019" / "hasoc2019.zip", 'r') as zipf:
        with io.TextIOWrapper(zipf.open("english_dataset/english_dataset.tsv", 'r'), encoding="utf-8") as fin:
            reader = csv.DictReader(fin, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            train_raw = list(reader)
        with io.TextIOWrapper(zipf.open("english_dataset/hasoc2019_en_test-2919.tsv", 'r'), encoding="utf-8") as fin:
            reader = csv.DictReader(fin, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            test_raw = list(reader)

    assert all(r['task_1'] in {'HOF', 'NOT'} for r in train_raw)
    assert all(r['task_1'] in {'HOF', 'NOT'} for r in test_raw)

    train_data = TrainDataset(
        name="hasoc2019",
        train_samples=[
            Sample(
                id=r['text_id'],
                text=r['text'],
                label=r['task_1'] == 'HOF',
            )
            for r in train_raw
        ],
        dev_samples=None,
    )
    test_data = TestDataset(
        name="hasoc2019",
        test_samples=[
            Sample(
                id=r['text_id'],
                text=r['text'],
                label=r['task_1'] == 'HOF',
            )
            for r in test_raw
        ],
    )

    return train_data, test_data
