
import csv
import io
import zipfile
from pathlib import Path

from aaai2023.datasets.classifier import Sample, TrainDataset, TestDataset


def prepare(
    raw: Path,
):
    file_stems = [
        "test",
        "test_labels",
        "train",
    ]

    parsed_files = {}
    with zipfile.ZipFile(raw / "jigsaw" / "jigsaw-toxic-comment-classification-challenge.zip", mode='r') as archive:
        for stem in file_stems:
            with zipfile.ZipFile(archive.open(f"{stem}.csv.zip")) as zipf:
                with io.TextIOWrapper(zipf.open(f"{stem}.csv", 'r'), encoding='utf-8') as fin:
                    reader = csv.DictReader(fin, delimiter=",", quoting=csv.QUOTE_MINIMAL)
                    parsed_files[stem] = list(reader)

    label_names = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    assert all(d[label] in {'0', '1'} for d in parsed_files['train'] for label in label_names)
    assert all(d[label] in {'-1','0', '1'} for d in parsed_files['test_labels'] for label in label_names)

    def dict_label(d):
        label_vals = [int(d[label]) for label in label_names]
        if any(l == -1 for l in label_vals):
            return -1
        elif any(l == 1 for l in label_vals):
            return 1
        else:
            return 0

    train_data = TrainDataset(
        name="jigsaw",
        train_samples=[
            Sample(
                id=f"jigsaw-{d['id']}",
                text=d['comment_text'],
                label=dict_label(d) == 1,
            )
            for d in parsed_files['train']
            if dict_label(d) != -1
        ],
        dev_samples=None,
    )

    test_map = {
        d['id']: d
        for d in parsed_files['test_labels']
    }

    test_data = TestDataset(
        name="jigsaw",
        test_samples=[
            Sample(
                id=f"jigsaw-{d['id']}",
                text=d['comment_text'],
                label=dict_label(test_map[d['id']]) == 1,
            )
            for d in parsed_files['test']
            if dict_label(test_map[d['id']]) != -1
        ]
    )

    return train_data, test_data
