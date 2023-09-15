
from pathlib import Path
import csv

from icwsm2024.datasets.classifier import Sample, TestDataset


def prepare(
    raw: Path,
):
    with (raw / "ethos" / "ethos.csv").open('r') as fin:
        reader = csv.DictReader(
            fin,
            delimiter=";",
            quoting=csv.QUOTE_MINIMAL,
        )
        raw = list(reader)

    test_data = TestDataset(
        name="ethos",
        test_samples=[
            Sample(
                id=f"ethos-{ix}",
                text=r['comment'],
                label=float(r['isHate']) >= .5,
            )
            for ix, r in enumerate(raw)
        ]
    )

    return None, test_data
