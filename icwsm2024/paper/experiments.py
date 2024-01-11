
from pathlib import Path
from dataclasses import asdict
import warnings
import json
from multiprocessing import get_context

from tqdm import tqdm

from icwsm2024.datasets.classifier import TestDataset, ScoredDataset
from icwsm2024.datasets.quantifier import (
    BinaryClassifierData,
    BinaryQuantificationData,
    QuantileUniform,
    SelectRandom,
    PlattScaling,
    Isotonic,
    HistogramBinning,
)
from icwsm2024.paper.util import (
    Experiment,
    ExperimentResult,
    QuantificationStrategy,
    SampleSelectionStrategy,
    is_subsampling,
)
from icwsm2024.paper.experiment_configurations import (
    compare_quantification_strategies,
    out_of_domain,
)


def build_clf_data(
    scores_file: str,
    test_folder: str,
) -> BinaryClassifierData:
    scores = ScoredDataset.load(Path(scores_file))
    if is_subsampling(scores.test_data):
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
    elif quant_strategy == "CPCC-ISO":
        return quant_data.calibrated_pcc(method=Isotonic())
    elif quant_strategy == "CPCC-HB10":
        return quant_data.calibrated_pcc(method=HistogramBinning(n=10))
    elif quant_strategy == "CPCC-HB100":
        return quant_data.calibrated_pcc(method=HistogramBinning(n=100))
    elif quant_strategy == "ABCC":
        return quant_data.bayesian_classify_and_count(agnostic=True)
    elif quant_strategy == "BCC":
        return quant_data.bayesian_classify_and_count(agnostic=False)
    elif quant_strategy == "Truth":
        return quant_data.true_prevalence()
    else:
        return f"unknown quantification strategy `{quant_strategy}`"


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
        quant_data = clf_data.split(
            n_dev=n_samples_to_select,
            selection_method=SelectRandom(seed=random_seed)
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
        quant_data = clf_data.split(
            n_dev=n_samples_to_select,
            selection_method=QuantileUniform(
                n_quantiles=n_quantiles,
                seed=random_seed,
            ),
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
    results_folder: Path,
    experiment_mode: str,
):
    if experiment_mode == "compare_quantification_strategies":
        exp_gen = compare_quantification_strategies(scores_folder=scores_folder)
    elif experiment_mode == "out_of_domain":
        exp_gen = out_of_domain(scores_folder=scores_folder)
    else:
        print(f"unknown experiment mode `{experiment_mode}`, exiting.")
        return

    results_file = results_folder / f"{experiment_mode}.jsonl"

    if results_file.exists():
        with results_file.open('r') as fin:
            already_done = {
                json.loads(line.strip())['hash_id']
                for line in fin
            }
    else:
        results_file.touch()
        already_done = set()

    exp_gen = [
        e
        for e in exp_gen
        if e.compute_db_hash() not in already_done
    ]

    fn = ExperimentWrapper(str(test_folder))

    pbar = tqdm(total=len(exp_gen))

    batch_size = 8192
    for start_ix in range(0, len(exp_gen), batch_size):
        next_batch = exp_gen[start_ix:start_ix+batch_size]
        with get_context("spawn").Pool() as pool:
            with results_file.open("a") as fout:
                for result in pool.imap_unordered(func=fn, iterable=next_batch, chunksize=128):
                    res_dict = asdict(result)
                    res_dict["hash_id"] = result.compute_db_hash()
                    fout.write(f"{json.dumps(res_dict)}\n")
                    pbar.update(1)
