
import gzip
import json
from typing import Tuple, List
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import numpy as np

from tqdm import tqdm

from aaai2023.datasets.classifier import (
    TrainDataset,
    TestDataset,
    ScoredDataset,
    ScoredSample,
)


def build_clf(
    ngram_range: Tuple[int, int],
    binary: bool,
    svm_c: float,
    seed: int = 0xdeadbeef,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                ngram_range=ngram_range,
                binary=binary,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,
                norm='l2',
            )),
            ("svm", LinearSVC(
                C=svm_c,
                class_weight='balanced',
                random_state=seed,
            )),
        ],
    )


def experiment(
    train: TrainDataset,
    tests: List[TestDataset],
    seed: int = 0xdeadbeef,
) -> List[ScoredDataset]:

    if train.dev_samples is not None:
        train_samples = train.train_samples
        dev_samples = train.dev_samples
    else:
        train_samples, dev_samples = train_test_split(
            train.train_samples,
            test_size=.1,
            random_state=seed,
            shuffle=True,
            stratify=[s.label for s in train.train_samples],
        )

    x_train = [s.text for s in train_samples]
    y_train = np.array([s.label for s in train_samples])

    x_dev = [s.text for s in dev_samples]
    y_dev = np.array([s.label for s in dev_samples])

    best_clf = None
    best_f1 = 0.

    for max_n in [2, 3, 5, 7]:
        for binary in [False, True]:
            for svm_c in [.001, .01, .1, 1., 10., 100., 1000.]:
                clf = build_clf(
                    ngram_range=(1, max_n),
                    binary=binary,
                    svm_c=svm_c,
                )
                clf.fit(X=x_train, y=y_train)
                y_pred = clf.predict(X=x_dev)

                f1 = f1_score(
                    y_true=y_dev,
                    y_pred=y_pred,
                    pos_label=1,
                    average='binary',
                )

                if f1 > best_f1:
                    best_clf = clf
                    best_f1 = f1

    best_params = best_clf.get_params()
    del best_params['memory']
    del best_params['steps']
    del best_params['tfidf']
    del best_params['svm']
    best_params['tfidf__dtype'] = str(best_params['tfidf_dtype'])

    result = []
    for test in tests:
        ids = [s.id for s in test.test_samples]
        x_test = [s.text for s in test.test_samples]
        ss = best_clf.decision_function(x_test)

        score_data = ScoredDataset(
            classifier_name="tfidf-svm",
            train_data=train.name,
            test_data=test.name,
            classifier_params=best_params,
            default_threshold=.0,
            scores=[
                ScoredSample(
                    id=i,
                    score=float(s),
                )
                for i, s in zip(ids, ss)
            ]
        )
        result.append(score_data)

    return result


def main(
    train: Path,
    test: Path,
    scores_path: Path,
):
    tests = []
    for test_f in test.glob("*.json.gz"):
        with gzip.open(test_f, 'rt') as fin:
            tests.append(TestDataset.from_json(json.load(fin)))

    for train_f in tqdm(train.glob("*.json.gz")):
        with gzip.open(train_f, 'rt') as fin:
            train_data = TrainDataset.from_json(json.load(fin))

        print(f"training on {train_data.name}")

        for o in experiment(train=train_data, tests=tests):
            with gzip.open(scores_path / f"{o.classifier_name}-{o.train_data}-{o.test_data}.json.gz", "wt") as fout:
                fout.write(json.dumps(obj=o.json(), indent=2))
                fout.write('\n')
