
import csv
from pathlib import Path

import numpy as np

from icwsm2024.datasets.classifier import Sample, TrainDataset, TestDataset


def prepare(
    raw: Path,
):
    with (raw / "ex_machina" / "comments.tsv").open('r') as fin:
        reader = csv.DictReader(fin, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        raw_comments = list(reader)

    with (raw / "ex_machina" / "annotations.tsv").open('r') as fin:
        reader = csv.DictReader(fin, delimiter="\t", quoting=csv.QUOTE_NONE)
        raw_anns = list(reader)

    annotations = {}
    for a in raw_anns:
        rev_id = a['rev_id']
        if annotations.get(rev_id) is None:
            annotations[rev_id] = []
        annotations[rev_id].append(float(a['attack']))

    splits = {
        split_name: [
            Sample(
                id=f"ex-machina-{c['rev_id']}",
                text=c['comment'].replace("NEWLINE_TOKEN", "\n"),
                label=bool(np.mean(annotations[c['rev_id']]) >= .5),
            )
            for c in raw_comments
            if c['split'] == split_name
        ]
        for split_name in ["train", "test", "dev"]
    }

    train_data = TrainDataset(
        name="ex-machina",
        train_samples=splits["train"],
        dev_samples=splits["dev"],
    )
    test_data = TestDataset(
        name="ex-machina",
        test_samples=splits['test'],
    )

    return train_data, test_data
