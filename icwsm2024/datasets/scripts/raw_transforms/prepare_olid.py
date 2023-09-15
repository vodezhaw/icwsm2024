
import csv
import io
import zipfile
from pathlib import Path

from icwsm2024.datasets.classifier import Sample, TrainDataset, TestDataset


def prepare(
    raw: Path,
):
    with zipfile.ZipFile(raw / "olid" / "OLIDv1.0.zip", 'r') as zipf:
        with io.TextIOWrapper(zipf.open("olid-training-v1.0.tsv", 'r'), encoding='utf-8') as fin:
            reader = csv.DictReader(fin, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            train_raw = list(reader)
        with io.TextIOWrapper(zipf.open("testset-levela.tsv", 'r'), encoding='utf-8') as fin:
            reader = csv.DictReader(fin, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            test_raw = list(reader)
        with io.TextIOWrapper(zipf.open("labels-levela.csv", 'r'), encoding='utf-8') as fin:
            reader = csv.DictReader(
                fin,
                delimiter=",",
                fieldnames=["id", "subtask_a"],
                quoting=csv.QUOTE_MINIMAL,
            )
            test_labels_raw = list(reader)

    assert all(d['subtask_a'] in {"OFF", "NOT"} for d in train_raw)
    assert all(d['subtask_a'] in {"OFF", "NOT"} for d in test_labels_raw)

    test_label_map = {
        d['id']: d['subtask_a']
        for d in test_labels_raw
    }

    train_data = TrainDataset(
        name="olid",
        train_samples=[
            Sample(
                id=f"olid-{d['id']}",
                text=d['tweet'],
                label=d['subtask_a'] == "OFF",
            )
            for d in train_raw
        ],
        dev_samples=None,
    )
    test_data = TestDataset(
        name="olid",
        test_samples=[
            Sample(
                id=f"olid-{d['id']}",
                text=d['tweet'],
                label=test_label_map[d['id']] == "OFF",
            )
            for d in test_raw
        ]
    )

    return train_data, test_data
