
from typing import Dict, Tuple
from pathlib import Path

from icwsm2024.datasets.classifier import (
    TrainDataset,
    TestDataset,
    ScoredDataset,
)


def load_train(train_path: Path) -> Dict[str, TrainDataset]:
    result = {}
    for train_f in train_path.glob('*.json.gz'):
        t = TrainDataset.load(train_f)
        result[t.name] = t
    return result


def load_test(test_path: Path) -> Dict[str, TestDataset]:
    result = {}
    for test_f in test_path.glob('*.json.gz'):
        t = TestDataset.load(test_f)
        result[t.name] = t
    return result


def load_scores(score_path: Path) -> Dict[Tuple[str, str, str], ScoredDataset]:
    result = {}
    for score_f in score_path.glob('*.json.gz'):
        s = ScoredDataset.load(score_f)
        result[s.classifier_name, s.train_data, s.test_data] = s
    return result
