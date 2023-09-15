
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from sklearn.metrics import roc_auc_score

from icwsm2024.datasets.hatespeech import load
from icwsm2024.paper.names import (
    TRAIN_DATASETS,
    TEST_DATASETS,
    DATASET_NAMES,
    CLF_FILE_STEMS,
)


def main(
    test_folder: Path,
    scores_folder: Path,
    out_folder: Path,
):
    paired = load(
        test=test_folder,
        scores=scores_folder,
    )

    env = Environment(loader=FileSystemLoader("icwsm2024/paper/templates/"))
    template = env.get_template("perf_table.tex")

    for clf in ["tfidf-svm", "google/electra-base-discriminator", "cardiffnlp/twitter-roberta-base"]:
        aucs = {
            DATASET_NAMES[train]: {
                DATASET_NAMES[test]: 0.
                for test in TEST_DATASETS
            }
            for train in TRAIN_DATASETS
        }

        rows = [DATASET_NAMES[train] for train in TRAIN_DATASETS]
        cols = [DATASET_NAMES[test] for test in TEST_DATASETS]

        for (clf_, train, test), b in paired.items():
            if clf_ != clf:
                continue

            if train in TRAIN_DATASETS and test in TEST_DATASETS:
                aucs[DATASET_NAMES[train]][DATASET_NAMES[test]] = roc_auc_score(
                    y_true=b.labels,
                    y_score=b.scores,
                    average=None,
                    multi_class='raise',
                )

        tex = template.render(
            train_datasets=rows,
            test_datasets=cols,
            auc=aucs,
        )

        with (out_folder / f"{CLF_FILE_STEMS[clf]}_rocauc.tex").open('w') as fout:
            fout.write(f"{tex}\n")

    aucs = {
        "dummy": {
            DATASET_NAMES[test]: 0.
            for test in TEST_DATASETS
        }
    }
    for (clf, train, test), b in paired.items():
        if clf != "perspective-api":
            continue
        assert train == "online-api"

        if test in TEST_DATASETS:
            aucs["dummy"][DATASET_NAMES[test]] = roc_auc_score(
                y_true=b.labels,
                y_score=b.scores,
                average=None,
                multi_class="raise",
            )
    tex = template.render(
        train_datasets=["dummy"],
        test_datasets=cols,
        auc=aucs,
    )
    with (out_folder / "perspective_rocauc.tex").open("w") as fout:
        fout.write(f'{tex}\n')


if __name__ == "__main__":
    main(
        test_folder=Path('./data/test'),
        scores_folder=Path("./data/scores"),
        out_folder=Path('./artefacts')
    )
