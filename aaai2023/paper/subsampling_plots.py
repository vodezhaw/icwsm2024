
from typing import Dict, Tuple, List
from pathlib import Path

from matplotlib import rc_context, pyplot as plt

import numpy as np

from aaai2023.paper.util import read_results, compute_errors, is_subsampling
from aaai2023.paper.names import SUBSAMPLING_SUFFIXES, SUBSAMPLING_PREVALENCES
from aaai2023.paper.plot_utils import all_box_plots


def main():
    res_path = Path('./data/results.jsonl')
    res = read_results(res_path)
    res = compute_errors(res)

    subsampling_domains = [
        "jigsaw",
        "ex-machina",
    ]

    # keep only in-domain experiments
    res = [
        r
        for r in res
        if r.other_domain_scores_file is None
    ]

    # keep only subsampling experiments
    res = [
        r
        for r in res
        if is_subsampling(Path(r.scores_file))
    ]

    # ignore errors
    res = [
        r
        for r in res
        if r.error_message is None
    ]

    # keep only 50 and 100 samplings
    res = [
        r
        for r in res
        if r.n_samples_to_select >= 50
    ]

    groupings = {}
    for domain in subsampling_domains:
        for suff in SUBSAMPLING_SUFFIXES:
            for q_strat in ['CC', "CPCC", "BCC"]:
                # jigsaw has no p10
                if domain == 'jigsaw' and suff == 'p10':
                    continue
                groupings[domain, f"{SUBSAMPLING_PREVALENCES[suff]:.3f}", q_strat] = [
                    r
                    for r in res
                    if r.quant_strategy == q_strat and
                       suff in r.scores_file and
                       domain in r.scores_file
                ]

    for domain in subsampling_domains:
        groups = {
            (prev, q): vs
            for (d, prev, q), vs in groupings.items()
        }
        all_box_plots(
            groupings=groups,
            x_labels=[
                f"{SUBSAMPLING_PREVALENCES[suff]:.3f}"
                for suff in (
                    SUBSAMPLING_SUFFIXES[:-1] if domain == 'jigsaw' else SUBSAMPLING_SUFFIXES
                )
            ],
            group_labels=['CC', 'CPCC', 'BCC'],
            group_style={
                'CC': {
                    'color': 'black',
                },
                'CPCC': {
                    'color': '#404749',
                    'linestyle': ':',
                },
                'BCC': {
                    'color': "#A9B0B3",
                    'linestyle': '--',
                },
            },
            file_name_fn=lambda clf, err: f"./artefacts/subsampling/{domain}-{clf}-{err}.png",
            x_label="Prevalence",
        )


if __name__ == '__main__':
    main()
