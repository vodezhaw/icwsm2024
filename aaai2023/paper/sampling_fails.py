
from pathlib import Path
from collections import Counter

import numpy as np

from tqdm import tqdm

from aaai2023.datasets.hatespeech import load
from aaai2023.datasets.quantifier import (
    Quantile,
    QuantileUniform,
    SelectRandom,
)
from aaai2023.paper.util import is_subsampling
from aaai2023.paper.names import RANDOM_SEEDS, SUBSAMPLING_PREVALENCES, SUBSAMPLING_SUFFIXES
from aaai2023.paper.plot_utils import bar_plots


def sample_sizes(
    scored_data,
    methods,
    method_style,
):
    scored_data = {
        (clf, train, test): data
        for (clf, train, test), data in scored_data.items()
        if not is_subsampling(test)
    }

    n_to_select = list(range(10, 201, 10))

    result = {}
    pbar = tqdm(total=len(methods) * len(n_to_select) * len(RANDOM_SEEDS))
    for method_name, m_fn in methods:
        for n in n_to_select:
            result[method_name, n] = Counter(
                data.split(n_dev=n, selection_method=m_fn(seed)).dev.labels.sum()
                for k, data in scored_data.items()
                for seed in RANDOM_SEEDS
            )
            pbar.update(len(RANDOM_SEEDS))

    percent_fails = {
        k: counts.get(0, 0) / sum(counts.values())
        for k, counts in result.items()
    }

    fails = {
        method: np.array([percent_fails[method, n] for n in n_to_select])
        for method, _ in methods
    }

    bar_plots(
        x_labels=list(map(str, n_to_select)),
        group_labels=[m for m, _ in methods],
        data=fails,
        group_style=method_style,
        x_label="N Selected",
        y_label="Fraction of Failures",
        save_as="./artefacts/fails/sample_size.png",
    )


def prevalence(
    scored_data,
    methods,
    method_style,
):
    scored_data = {
        (clf, train, test): data
        for (clf, train, test), data in scored_data.items()
        if is_subsampling(test)
    }

    pbar = tqdm(total=len(methods) * len(SUBSAMPLING_SUFFIXES) * len(RANDOM_SEEDS))
    result = {}
    for method_name, m_fn in methods:
        for suff in SUBSAMPLING_SUFFIXES:
            result[method_name, suff] = Counter(
                data.split(n_dev=100, selection_method=m_fn(seed)).dev.labels.sum()
                for (_, _, test), data in scored_data.items()
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

    bar_plots(
        x_labels=[f"{SUBSAMPLING_PREVALENCES[suff]:.3f}" for suff in SUBSAMPLING_SUFFIXES],
        group_labels=[m for m, _ in methods],
        data=fail_data,
        group_style=method_style,
        x_label="Prevalence",
        y_label="Fraction of Failures",
        save_as="./artefacts/fails/prevalence.png",
    )


def main():
    test_folder = Path('./data/test')
    scores_folder = Path('./data/scores')

    scored_data = load(
        test=test_folder,
        scores=scores_folder,
    )

    methods = [
        ("random", lambda s: SelectRandom(seed=s)),
        # ("quantile", lambda s: Quantile(n_quantiles=10, seed=s)),
        ("quantile", lambda s: QuantileUniform(n_quantiles=10, seed=s)),
    ]

    method_style = {
        "random": {'color': 'black', 'fill': False, 'hatch': "/"},
        "quantile": {'color': 'black', 'fill': False, 'hatch': "."},
    }

    sample_sizes(scored_data, methods, method_style)
    prevalence(scored_data, methods, method_style)


if __name__ == '__main__':
    main()
