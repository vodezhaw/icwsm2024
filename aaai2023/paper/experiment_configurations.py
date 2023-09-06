
from typing import Iterator
from pathlib import Path

from tqdm import tqdm

from aaai2023.datasets.classifier import ScoredDataset
from aaai2023.paper.util import (
    Experiment,
    is_subsampling,
)
from aaai2023.paper.names import (
    QUANTIFICATION_STRATEGIES,
    SELECTION_STRATEGIES,
    RANDOM_SEEDS,
)


def compare_quantification_strategies(
    scores_folder: Path,
) -> Iterator[Experiment]:
    for score_file in scores_folder.glob("*.json.gz"):
        if is_subsampling(score_file):
            continue

        for qstrat in QUANTIFICATION_STRATEGIES + ['Truth']:
            if qstrat in {'PCC', "PACC"} and "tfidf-svm" in score_file.name:
                continue
            for sstrat in SELECTION_STRATEGIES:
                for seed in RANDOM_SEEDS:
                    yield Experiment(
                        scores_file=str(score_file),
                        sample_selection_strategy=sstrat,
                        quant_strategy=qstrat,
                        n_samples_to_select=100,  # baseline N
                        n_quantiles=10 if qstrat == "quantile" else None,
                        other_domain_scores_file=None,
                        random_seed=seed,
                    )


def out_of_domain(
    scores_folder: Path,
) -> Iterator[Experiment]:
    score_infos = []
    print('load score data')
    for score_file in tqdm(list(scores_folder.glob("*.json.gz"))):
        if is_subsampling(score_file):
            continue
        data = ScoredDataset.load(score_file)
        score_infos.append({
            "score_file": score_file,
            "clf": data.classifier_name,
            "train": data.train_data,
            "test": data.test_data,
        })

    for info in score_infos:

        for other_info in score_infos:
            same_clf = info['clf'] == other_info['clf']
            same_train = info['train'] == other_info['train']
            same_test = info['test'] == other_info['test']

            if same_clf and same_train and (not same_test):

                for qstrat in QUANTIFICATION_STRATEGIES + ['Truth']:
                    if qstrat in {"PCC", "PACC"} and "tfidf-svm" in info['clf']:
                        continue
                    for seed in RANDOM_SEEDS:
                        yield Experiment(
                            scores_file=str(info['score_file']),
                            sample_selection_strategy="other-domain",
                            quant_strategy=qstrat,
                            n_samples_to_select=None,
                            n_quantiles=None,
                            other_domain_scores_file=str(other_info['score_file']),
                            random_seed=seed,
                        )
