
from pathlib import Path
import csv

from icwsm2024.datasets.classifier import Sample, TestDataset


def prepare(
    raw: Path,
):
    with (raw / "swad" / "swad.tsv").open('r') as fin:
        reader = csv.DictReader(
            fin,
            delimiter='\t',
            fieldnames=["id", "tweet", "label"],
            quoting=csv.QUOTE_NONE,
        )
        raw = list(reader)

    assert all(d['label'] in {'Yes', 'No'} for d in raw)

    test_data = TestDataset(
        name="swad",
        test_samples=[
            Sample(
                id=f"swad-{d['id']}",
                text=d['tweet'],
                label=d['label'] == 'Yes',
            )
            for d in raw
        ],
    )

    return None, test_data
