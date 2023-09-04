
from typing import List
from pathlib import Path
import json
from dataclasses import dataclass, asdict

from aaai2023.scripts.experiments import ExperimentResult
from aaai2023.paper.names import SUBSAMPLING_SUFFIXES


@dataclass(frozen=True)
class ExperimentError(ExperimentResult):
    true_prevalence: float | None = None
    absolute_error: float | None = None
    absolute_percentage_error: float | None = None


def ae(p_pred: float | None, p_true: float | None) -> float | None:
    if p_pred is None or p_true is None:
        return None
    return abs(p_pred - p_true)


def ape(p_pred: float, p_true: float) -> float | None:
    if p_pred is None or p_true is None:
        return None
    return abs(p_pred - p_true) / p_true


def read_results(res_file: Path) -> List[ExperimentResult]:
    res = []
    with res_file.open('r') as fin:
        for line in fin:
            d = json.loads(line.strip())
            res.append(ExperimentResult(
                scores_file=d['scores_file'],
                sample_selection_strategy=d['sample_selection_strategy'],
                quant_strategy=d['quant_strategy'],
                n_samples_to_select=d['n_samples_to_select'],
                n_quantiles=d['n_quantiles'],
                other_domain_scores_file=d['other_domain_scores_file'],
                random_seed=d['random_seed'],
                predicted_prevalence=d['predicted_prevalence'],
                error_message=d['error_message'],
            ))
    return res


def is_subsampling(name: Path | str) -> bool:
    if issubclass(type(name), Path):
        name = name.name.split('.')[0]
    return any(name.endswith(f"-{suffix}") for suffix in SUBSAMPLING_SUFFIXES)


def compute_errors(exps: List[ExperimentResult]) -> List[ExperimentError]:

    def exp_key(e: ExperimentResult):
        return (
            e.scores_file,
            e.sample_selection_strategy,
            e.n_samples_to_select,
            e.n_quantiles,
            e.other_domain_scores_file,
            e.random_seed,
        )

    truths = {
        exp_key(e): e.predicted_prevalence
        for e in exps
        if e.quant_strategy == 'Truth'
    }

    return [
        ExperimentError(
            **asdict(e),
            true_prevalence=truths[exp_key(e)],
            absolute_error=ae(
                p_pred=e.predicted_prevalence,
                p_true=truths[exp_key(e)],
            ),
            absolute_percentage_error=ape(
                p_pred=e.predicted_prevalence,
                p_true=truths[exp_key(e)],
            ),
        )
        for e in exps
    ]
