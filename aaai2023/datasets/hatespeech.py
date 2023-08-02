
import gzip
import json
from pathlib import Path

from aaai2023.datasets.classifier import TestDataset, ScoredDataset
from aaai2023.datasets.quantifier import BinaryClassifierData


def load(
    test: Path,
    scores: Path,
):
    test_sets = {}
    for test_file in test.glob('*.json.gz'):
        with gzip.open(test_file, 'rt') as fin:
            t = TestDataset.from_json(json.load(fin))
        test_sets[t.name] = t

    score_sets = {}
    for score_file in scores.glob("*.json.gz"):
        with gzip.open(score_file, 'rt') as fin:
            ss = ScoredDataset.from_json(json.load(fin))
        score_sets[(ss.classifier_name, ss.train_data, ss.test_data)] = ss

    return {
        (clf, train_set, test_set): BinaryClassifierData.from_test_scores(
            test_set=test_sets[test_set],
            scores=ss,
        )
        for (clf, train_set, test_set), ss in score_sets.items()
    }
