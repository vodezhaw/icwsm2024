
from pathlib import Path
import json
import gzip

from icwsm2024.datasets.scripts.raw_transforms.prepare_davidson import (
    prepare_hate_off as davidson_hate_off,
    prepare_hate_only as davidson_hate_only,
)
from icwsm2024.datasets.scripts.raw_transforms.prepare_dgh import prepare as dgh
from icwsm2024.datasets.scripts.raw_transforms.prepare_ethos import prepare as ethos
from icwsm2024.datasets.scripts.raw_transforms.prepare_ex_machina import prepare as ex_machina
from icwsm2024.datasets.scripts.raw_transforms.prepare_hasoc import prepare as hasoc
from icwsm2024.datasets.scripts.raw_transforms.prepare_hateval import prepare as hateval
from icwsm2024.datasets.scripts.raw_transforms.prepare_jigsaw import prepare as jigsaw
from icwsm2024.datasets.scripts.raw_transforms.prepare_olid import prepare as olid
from icwsm2024.datasets.scripts.raw_transforms.prepare_solid import prepare as solid
from icwsm2024.datasets.scripts.raw_transforms.prepare_swad import prepare as swad
from icwsm2024.datasets.scripts.raw_transforms.prepare_wassa import prepare as wassa


def main(
    raw: Path,
    train: Path,
    test: Path,
):
    datasets = [
        davidson_hate_off,
        davidson_hate_only,
        dgh,
        ethos,
        ex_machina,
        hasoc,
        hateval,
        jigsaw,
        olid,
        solid,
        swad,
        wassa,
    ]

    for d in datasets:
        train_data, test_data = d(raw)
        if train_data is not None:
            with gzip.open(train / f"{train_data.name}.json.gz", "wt") as fout:
                fout.write(json.dumps(obj=train_data.json(), indent=2))
                fout.write('\n')

        if test_data is not None:
            with gzip.open(test / f"{test_data.name}.json.gz", "wt") as fout:
                fout.write(json.dumps(obj=test_data.json(), indent=2))
                fout.write('\n')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", dest="raw", required=True, type=Path)
    parser.add_argument("--train", dest="train", required=True, type=Path)
    parser.add_argument("--test", dest="test", required=True, type=Path)
    args = parser.parse_args()

    main(
        raw=args.raw,
        train=args.train,
        test=args.test,
    )
