
from typing import Literal, Iterator, Iterable
from pathlib import Path
from dataclasses import dataclass, asdict
from multiprocessing import get_context
import warnings
import hashlib
import json

import numpyro
from tqdm import tqdm

from aaai2023.datasets.classifier import TestDataset, ScoredDataset
from aaai2023.datasets.quantifier import (
    BinaryClassifierData,
    BinaryQuantificationData,
)


SEEDS = [
    0xdeadbeef,
    0xbeefbabe,
    0xcafebabe,
    0xb00b00b0,
    0xafed00d3,
]

SampleSelectionStrategy = Literal[
    "random",
    "quantile",
    "other-domain",
]

QuantificationStrategy = Literal[
    "CC",
    "ACC",
    "PCC",
    "PACC",
    "CPCC",
    "ABCC",
    "BCC",
    "Truth",
]


def build_clf_data(
    scores_file: str,
    test_folder: str,
) -> BinaryClassifierData:
    scores = ScoredDataset.load(Path(scores_file))
    if any(scores.test_data.endswith(suffix) for suffix in {"-p01", "-p05", "-p10", "-p1", "-p3", "-p5"}):
        test_name = "-".join(scores.test_data.split("-")[:-1])
    else:
        test_name = scores.test_data
    test = TestDataset.load(Path(test_folder) / f"{test_name}.json.gz")
    return BinaryClassifierData.from_test_scores(
        test_set=test,
        scores=scores,
    )


def quantify(
    quant_strategy: QuantificationStrategy,
    quant_data: BinaryQuantificationData,
) -> float | str:
    if quant_strategy == "CC":
        return quant_data.classify_and_count()
    elif quant_strategy == "ACC":
        return quant_data.adjusted_classify_and_count()
    elif quant_strategy == "PCC":
        return quant_data.probabilistic_classify_and_count()
    elif quant_strategy == "PACC":
        return quant_data.probabilistic_adjusted_classify_and_count()
    elif quant_strategy == "CPCC":
        return quant_data.calibrated_pcc()
    elif quant_strategy == "ABCC":
        return quant_data.bayesian_classify_and_count(agnostic=True)
    elif quant_strategy == "BCC":
        return quant_data.bayesian_classify_and_count(agnostic=False)
    elif quant_strategy == "Truth":
        return quant_data.true_prevalence()
    else:
        return f"unknown quantification strategy `{quant_strategy}`"


@dataclass(frozen=True)
class Experiment:
    scores_file: str
    sample_selection_strategy: SampleSelectionStrategy
    quant_strategy: QuantificationStrategy
    n_samples_to_select: int | None = None
    n_quantiles: int | None = None
    other_domain_scores_file: str | None = None
    random_seed: int | None = None

    def compute_db_hash(self) -> str:
        string_repr = (f"{self.scores_file}-"
                       f"{self.sample_selection_strategy}-"
                       f"{self.quant_strategy}-"
                       f"{self.n_samples_to_select}-"
                       f"{self.n_quantiles}-"
                       f"{self.other_domain_scores_file}-"
                       f"{self.random_seed}")
        h = hashlib.new("sha256")
        h.update(string_repr.encode("utf-8"))
        return h.hexdigest()


@dataclass(frozen=True)
class ExperimentResult(Experiment):
    predicted_prevalence: float | None = None
    error_message: str | None = None


def enumerate_random(n_samples: Iterable[int], random_seeds: Iterable[int]):
    for n in n_samples:
        for seed in random_seeds:
            yield "random", n, None, None, seed


def enumerate_quantiled(
    n_samples: Iterable[int],
    n_quantiles: Iterable[int],
    random_seeds: Iterable[int],
):
    for n in n_samples:
        for n_quant in n_quantiles:
            for seed in random_seeds:
                yield "quantile", n, n_quant, None, seed


def enumerate_other(
    clf: str,
    train: str,
    test: str,
    infos: dict,
):
    for of, info in infos.items():
        if info['clf'] == clf and info['train'] == train and info['test'] != test:
            yield "other-domain", None, None, of, None


def enumerate_experiments(
    scores_folder: Path,
    n_samples: Iterable[int] = (10, 25, 50, 100),
    n_quantiles: Iterable[int] = (5, 10),
    random_seeds: Iterable[int] = tuple(SEEDS),
) -> Iterator[Experiment]:
    score_information = {}
    for sf in scores_folder.glob("*.json.gz"):
        data = ScoredDataset.load(sf)
        score_information[str(sf)] = {
            "clf": data.classifier_name,
            "train": data.train_data,
            "test": data.test_data,
            "default_thresh": data.default_threshold,
        }
        del data

    for sf, info in score_information.items():
        for qstrat in ["CC", "ACC", "PCC", "PACC", "CPCC", "ABCC", "BCC", "Truth"]:
            if qstrat in {"PCC", "PACC"} and info["default_thresh"] != .5:
                continue

            for gen in [
                enumerate_random(n_samples=n_samples, random_seeds=random_seeds),
                enumerate_quantiled(
                    n_samples=n_samples,
                    n_quantiles=n_quantiles,
                    random_seeds=random_seeds,
                ),
                enumerate_other(
                    clf=info['clf'],
                    train=info['train'],
                    test=info['train'],
                    infos=score_information,
                ),
            ]:
                for sel, ns, nq, of, seed in gen:
                    yield Experiment(
                        scores_file=sf,
                        sample_selection_strategy=sel,
                        quant_strategy=qstrat,
                        n_samples_to_select=ns,
                        n_quantiles=nq,
                        other_domain_scores_file=of,
                        random_seed=seed,
                    )


def experiment(
    test_folder: str,
    scores_file: str,
    sample_selection_strategy: SampleSelectionStrategy,
    quant_strategy: QuantificationStrategy,
    n_samples_to_select: int | None = None,
    n_quantiles: int | None = None,
    other_domain_scores_file: str | None = None,
    random_seed: int | None = None,
) -> float | str:

    clf_data = build_clf_data(
        scores_file=scores_file,
        test_folder=test_folder,
    )

    if sample_selection_strategy == "random":
        if n_samples_to_select is None:
            return (f"need to provide `n_samples_to_select` "
                    f"for selection strategy `random`")
        if random_seed is None:
            return (f"need to provide `random_seed` "
                    f"for selection strategy `random`")
        quant_data = clf_data.random_split(
            n_dev=n_samples_to_select,
            n_quantiles=None,
            random_state=random_seed
        )
    elif sample_selection_strategy == "quantile":
        if n_samples_to_select is None:
            return (f"need to provide `n_samples_to_select` "
                    f"for selection strategy `quantile`")
        if random_seed is None:
            return (f"need to provide `random_seed` "
                    f"for selection strategy `quantile`")
        if n_quantiles is None:
            return (f"need to provide `n_quantiles` "
                    f"for selection strategy `quantile`")
        quant_data = clf_data.random_split(
            n_dev=n_samples_to_select,
            n_quantiles=n_quantiles,
            random_state=random_seed,
        )
    elif sample_selection_strategy == "other-domain":
        if other_domain_scores_file is None:
            return (f"need to provide `other_domain_scores_file` "
                    f"for selection strategy `other-domain`")
        dev_data = build_clf_data(
            scores_file=other_domain_scores_file,
            test_folder=test_folder,
        )
        quant_data = BinaryQuantificationData(
            dev=dev_data,
            test=clf_data,
        )
    else:
        return f"unknown selection strategy `{sample_selection_strategy}`"

    # our selection strategy did not select any positive samples
    if quant_data.dev.labels.sum() == 0:
        return "sample selection strategy did not produce positive samples"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(quantify(
                quant_strategy=quant_strategy,
                quant_data=quant_data,
            ))
    except Exception as e:
        return f"failed with Exception:\n{e}"


class ExperimentWrapper:

    def __init__(self, test_folder: str):
        self.test_folder = test_folder

    def __call__(self, exp: Experiment) -> ExperimentResult:
        res = experiment(test_folder=self.test_folder, **asdict(exp))
        if type(res) is str:
            return ExperimentResult(
                predicted_prevalence=None,
                error_message=res,
                **asdict(exp)
            )
        elif type(res) is float:
            return ExperimentResult(
                predicted_prevalence=res,
                error_message=None,
                **asdict(exp)
            )
        else:
            raise ValueError(f"unexpected return type `{type(res)}`")


def run_all(
    test_folder: Path,
    scores_folder: Path,
    results_file: Path,
):
    if results_file.exists():
        with results_file.open('r') as fin:
            already_done = {
                json.loads(line.strip())['hash_id']
                for line in fin
            }
    else:
        results_file.touch()
        already_done = set()

    fn = ExperimentWrapper(str(test_folder))

    with get_context("spawn").Pool(
        processes=None,
        maxtasksperchild=8192,
    ) as pool:
        exp_gen = tqdm([
            e
            for e in enumerate_experiments(
                scores_folder=scores_folder,
            )
            if e.compute_db_hash() not in already_done
        ])
        with results_file.open("a") as fout:
            for result in pool.imap_unordered(func=fn, iterable=exp_gen, chunksize=128):
                res_dict = asdict(result)
                res_dict["hash_id"] = result.compute_db_hash()
                fout.write(json.dumps(res_dict))
                fout.write("\n")
