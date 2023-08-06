
from pathlib import Path

from aaai2023.datasets.quantifier import BinaryClassifierData
from aaai2023.datasets import load_test, load_scores


def load(
    test: Path,
    scores: Path,
):
    test_sets = load_test(test)
    score_sets = load_scores(scores)
    return {
        (clf, train_set, test_set): BinaryClassifierData.from_test_scores(
            test_set=test_sets[test_set],
            scores=ss,
        )
        for (clf, train_set, test_set), ss in score_sets.items()
    }
