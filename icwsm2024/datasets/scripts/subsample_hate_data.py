
from pathlib import Path

from sklearn.model_selection import train_test_split

import numpy as np

from tqdm import tqdm

from icwsm2024.datasets import load_test, load_scores
from icwsm2024.datasets.classifier import TestDataset, ScoredDataset

SHORT_NAMES = {
    "google/electra-base-discriminator": "electra",
    "cardiffnlp/twitter-roberta-base": "cardiffnlp",
}


def main(
    test_path: Path,
    scores_path: Path,
):
    tests = load_test(test_path)
    scores = load_scores(scores_path)

    p_data = [
        (0.01, "1"),
        (0.02, "2"),
        (0.03, "3"),
        (0.05, "5"),
        (0.07, "7"),
        (.1, "10"),
    ]

    scores = {
        test_set: [
            ss
            for (_, _, test_), ss in scores.items()
            if test_ == test_set
        ]
        for test_set in tests.keys()
    }

    for test_name, test_set in tests.items():
        p_orig = np.mean([s.label for s in test_set.test_samples])

        pos_ids = [
            s.id
            for s in test_set.test_samples
            if s.label
        ]
        neg_ids = [
            s.id
            for s in test_set.test_samples
            if not s.label
        ]
        n_neg = len(neg_ids)

        for p, p_name in tqdm(p_data):
            if p >= p_orig:
                continue
            new_n_pos = int(.5 + n_neg * (p / (1. - p)))

            if new_n_pos < 15:
                continue

            # this shouldnt happen, but doesnt hurt to check
            if new_n_pos > len(pos_ids):
                continue

            if new_n_pos == len(pos_ids):
                pos_keep = pos_ids
            else:
                pos_keep, _ = train_test_split(
                    pos_ids,
                    train_size=new_n_pos,
                    random_state=0xdeadbeef,
                    shuffle=True,
                )

            new_id_set = set(neg_ids).union(set(pos_keep))

            new_test_set = TestDataset(
                name=f"{test_set.name}-p{p_name}",
                test_samples=[
                    s
                    for s in test_set.test_samples
                    if s.id in new_id_set
                ]
            )
            new_test_set.save(test_path / f"{new_test_set.name}.json.gz")

            for scores_data in tqdm(scores[test_name]):
                new_scores_data = ScoredDataset(
                    classifier_name=scores_data.classifier_name,
                    train_data=scores_data.train_data,
                    test_data=new_test_set.name,
                    classifier_params=scores_data.classifier_params,
                    default_threshold=scores_data.default_threshold,
                    scores=[
                        s
                        for s in scores_data.scores
                        if s.id in new_id_set
                    ]
                )
                name = new_scores_data.classifier_name
                name = SHORT_NAMES.get(name, name)
                new_scores_data.save(scores_path / f"{name}-{new_scores_data.train_data}-{new_scores_data.test_data}.json.gz")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", dest="test", type=Path, required=True)
    parser.add_argument(
        "--scores", dest="scores", type=Path, required=True)
    args = parser.parse_args()

    main(
        test_path=args.test,
        scores_path=args.scores,
    )
