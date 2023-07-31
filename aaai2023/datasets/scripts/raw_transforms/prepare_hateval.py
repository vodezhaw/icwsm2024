
from pathlib import Path
import zipfile
import csv
import io

from aaai2023.datasets.classifier import Sample, TrainDataset, TestDataset


def prepare(
    raw: Path,
):

    read_success = False

    while not read_success:
        print("password for HatEval 2019 data:")
        pwd = input()
        pwd = pwd.strip()
        try:
            with zipfile.ZipFile(raw / "hateval2019" / "hateval2019.zip", mode='r') as zipf:
                with io.TextIOWrapper(zipf.open('hateval2019_en_train.csv', mode='r', pwd=pwd.encode('utf-8')), encoding='utf-8') as fin:
                    reader = csv.DictReader(fin, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    train_raw = list(reader)
                with io.TextIOWrapper(zipf.open('hateval2019_en_dev.csv', mode='r', pwd=pwd.encode('utf-8')), encoding='utf-8') as fin:
                    reader = csv.DictReader(fin, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    dev_raw = list(reader)
                with io.TextIOWrapper(zipf.open('hateval2019_en_test.csv', mode='r', pwd=pwd.encode('utf-8')), encoding='utf-8') as fin:
                    reader = csv.DictReader(fin, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    test_raw = list(reader)
        except RuntimeError:
            print("Could not read zipfile")
            read_success = False
            continue

        read_success = True

    assert all(r['HS'] in {'0', '1'} for r in train_raw)
    assert all(r['HS'] in {'0', '1'} for r in dev_raw)
    assert all(r['HS'] in {'0', '1'} for r in test_raw)

    train_samples = [
        Sample(
            id=f"hateval2019-{r['id']}",
            text=r['text'],
            label=r['HS'] == '1',
        )
        for r in train_raw
    ]
    dev_samples = [
        Sample(
            id=f"hateval2019-{r['id']}",
            text=r['text'],
            label=r['HS'] == '1',
        )
        for r in dev_raw
    ]
    test_samples = [
        Sample(
            id=f"hateval2019-{r['id']}",
            text=r['text'],
            label=r['HS'] == '1',
        )
        for r in test_raw
    ]

    train_data = TrainDataset(
        name="hateval2019",
        train_samples=train_samples,
        dev_samples=dev_samples,
    )
    test_data = TestDataset(
        name="hateval2019",
        test_samples=test_samples,
    )
    return train_data, test_data
