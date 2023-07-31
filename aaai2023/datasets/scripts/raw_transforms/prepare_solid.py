
import zipfile
import io
import csv
from pathlib import Path

from aaai2023.datasets.classifier import Sample, TestDataset


def prepare(
    raw: Path
):
    with zipfile.ZipFile(raw / "solid" / "extended_test-20200717T190516Z-001.zip", 'r') as zipf:
        with io.TextIOWrapper(zipf.open("extended_test/test_a_tweets_all.tsv", 'r'), encoding='utf-8') as fin:
            reader = csv.DictReader(fin, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            test_raw = list(reader)
        with io.TextIOWrapper(zipf.open("extended_test/test_a_labels_all.csv", 'r'), encoding='utf-8') as fin:
            reader = csv.DictReader(
                fin,
                delimiter=',',
                fieldnames=['id', 'label'],
                quoting=csv.QUOTE_MINIMAL,
            )
            test_labels_raw = list(reader)

    assert all(d['label'] in {'OFF', 'NOT'} for d in test_labels_raw)

    test_label_map = {
        d['id']: d['label']
        for d in test_labels_raw
    }
    test_data = TestDataset(
        name="solid",
        test_samples=[
            Sample(
                id=f"solid-{d['id']}",
                text=d['tweet'],
                label=test_label_map[d['id']] == "OFF",
            )
            for d in test_raw
            if d['id'] in test_label_map.keys()
        ]
    )

    return None, test_data
