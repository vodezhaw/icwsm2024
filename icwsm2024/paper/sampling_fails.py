import json
from pathlib import Path
from collections import Counter

import numpy as np

from tqdm import tqdm

from icwsm2024.datasets.hatespeech import load
from icwsm2024.datasets.quantifier import (
    QuantileUniform,
    SelectRandom,
)
from icwsm2024.paper.util import is_subsampling
from icwsm2024.paper.names import RANDOM_SEEDS, SUBSAMPLING_PREVALENCES, SUBSAMPLING_SUFFIXES
from icwsm2024.paper.plot_utils import bar_plots


def sample_sizes(
    data,
    methods,
    method_style,
):
    data = {
        test: d
        for test, d in data.items()
        if not is_subsampling(test)
    }

    n_to_select = list(range(10, 121, 10))

    result = {}
    pbar = tqdm(total=len(methods) * len(n_to_select) * len(RANDOM_SEEDS))
    for method_name, m_fn in methods:
        for n in n_to_select:
            result[method_name, n] = Counter(
                bin_data.split(n_dev=n, selection_method=m_fn(seed)).dev.labels.sum()
                for test, bin_data in data.items()
                for seed in RANDOM_SEEDS
            )
            pbar.update(len(RANDOM_SEEDS))

    percent_fails = {
        k: counts.get(0, 0) / sum(counts.values())
        for k, counts in result.items()
    }

    with open('paper_plots/fails/sample_size_raw.json', 'w') as fout:
        json.dump(
            fp=fout,
            obj={
                "__".join(map(str, k)): v
                for k, v in percent_fails.items()
            },
            indent=2,
        )

    fails = {
        method: np.array([percent_fails[method, n] for n in n_to_select])
        for method, _ in methods
    }

    bar_plots(
        x_labels=list(map(str, n_to_select)),
        group_labels=[m for m, _ in methods],
        data=fails,
        group_style=method_style,
        x_label="$N_{calib}$",
        y_label="Fraction of Failures",
        save_as="./paper_plots/fails/sample_size.png",
    )


def prevalence(
    data,
    methods,
    method_style,
):
    data = {
        test: d
        for test, d in data.items()
        if is_subsampling(test)
    }

    pbar = tqdm(total=len(methods) * len(SUBSAMPLING_SUFFIXES) * len(RANDOM_SEEDS))
    result = {}
    for method_name, m_fn in methods:
        for suff in SUBSAMPLING_SUFFIXES:
            result[method_name, suff] = Counter(
                int(bin_data.split(n_dev=100, selection_method=m_fn(seed)).dev.labels.sum())
                for test, bin_data in data.items()
                for seed in RANDOM_SEEDS
                if test.endswith(suff)
            )
            pbar.update(len(RANDOM_SEEDS))

    fail_data = {
        method: np.array([
            result[method, suff].get(0, 0) / sum(result[method, suff].values())
            for suff in SUBSAMPLING_SUFFIXES
        ])
        for method, _ in methods
    }

    with open('paper_plots/fails/prevalence_raw.json', 'w') as fout:
        json.dump(
            fp=fout,
            obj={
                "__".join(map(str, k)): v.get(0, 0) / sum(result[k].values())
                for k, v in result.items()
            },
            indent=2,
        )

    bar_plots(
        x_labels=[f"{SUBSAMPLING_PREVALENCES[suff]:.3f}" for suff in SUBSAMPLING_SUFFIXES],
        group_labels=[m for m, _ in methods],
        data=fail_data,
        group_style=method_style,
        x_label="Prevalence",
        y_label="Fraction of Failures",
        save_as="./paper_plots/fails/prevalence.png",
    )


def main():
    test_folder = Path('./data/test')
    scores_folder = Path('./data/scores')

    scored_data = load(
        test=test_folder,
        scores=scores_folder,
    )

    only_one_test = {}
    for (_, _, test), d in scored_data.items():
        if only_one_test.get(test) is None:
            only_one_test[test] = d

    methods = [
        ("random", lambda s: SelectRandom(seed=s)),
        # ("quantile", lambda s: Quantile(n_quantiles=10, seed=s)),
        ("quantile", lambda s: QuantileUniform(n_quantiles=10, seed=s)),
    ]

    method_style = {
        "random": {'color': 'black', 'fill': False, 'hatch': "/"},
        "quantile": {'color': 'black', 'fill': False, 'hatch': "."},
    }

    sample_sizes(only_one_test, methods, method_style)
    prevalence(only_one_test, methods, method_style)


if __name__ == '__main__':
    main()
