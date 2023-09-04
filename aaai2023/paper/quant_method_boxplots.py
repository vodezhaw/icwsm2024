
from pathlib import Path

import numpy as np

from aaai2023.paper.util import read_results, is_subsampling, compute_errors
from aaai2023.paper.plot_utils import box_plots
from aaai2023.paper.names import QUANTIFICATION_STRATEGIES, SELECTION_STRATEGIES


def main():
    res_path = Path('./data/results.jsonl')
    res = read_results(res_path)
    res = compute_errors(res)

    # keep only in-domain experiments
    res = [
        r
        for r in res
        if r.other_domain_scores_file is None
    ]

    # ignore sub-sampling experiments
    res = [
        r
        for r in res
        if not is_subsampling(Path(r.scores_file))
    ]

    # ignore errors
    res = [
        r
        for r in res
        if r.error_message is None
    ]

    groupings = {}
    for q_strat in QUANTIFICATION_STRATEGIES:
        for s_strat in SELECTION_STRATEGIES:
            groupings[q_strat, s_strat] = [
                e
                for e in res
                if e.sample_selection_strategy == s_strat and e.quant_strategy == q_strat
            ]

    aes = {
        k: np.array([e.absolute_error for e in vs])
        for k, vs in groupings.items()
    }

    apes = {
        k: np.array([e.absolute_percentage_error for e in vs])
        for k, vs in groupings.items()
    }

    box_plots(
        x_labels=QUANTIFICATION_STRATEGIES,
        group_labels=SELECTION_STRATEGIES,
        data=aes,
        group_style={
            'random': {
                'color': 'black',
            },
            'quantile': {
                'color': "#A9B0B3",
                'linestyle': '--',
            }
        },
        y_label="Absolute Error",
        save_as="./artefacts/qstrats/all_AE.png",
    )
    box_plots(
        x_labels=QUANTIFICATION_STRATEGIES,
        group_labels=SELECTION_STRATEGIES,
        data=apes,
        group_style={
            'random': {
                'color': 'black',
            },
            'quantile': {
                'color': "#A9B0B3",
                'linestyle': '--',
            }
        },
        y_label="Absolute Relative Error",
        save_as="./artefacts/qstrats/all_APE.png",
    )

    for clf in ['electra', 'cardiffnlp', 'tfidf-svm', 'perspective']:
        aes = {
            k: np.array([
                e.absolute_error
                for e in vs
                if Path(e.scores_file).stem.startswith(clf)
            ])
            for k, vs in groupings.items()
        }
        apes = {
            k: np.array([
                e.absolute_percentage_error
                for e in vs
                if Path(e.scores_file).stem.startswith(clf)
            ])
            for k, vs in groupings.items()
        }
        box_plots(
            x_labels=QUANTIFICATION_STRATEGIES,
            group_labels=SELECTION_STRATEGIES,
            data=aes,
            group_style={
                'random': {
                    'color': 'black',
                },
                'quantile': {
                    'color': "#A9B0B3",
                    'linestyle': '--',
                }
            },
            y_label="Absolute Error",
            save_as=f"./artefacts/qstrats/{clf}_AE.png",
        )
        box_plots(
            x_labels=QUANTIFICATION_STRATEGIES,
            group_labels=SELECTION_STRATEGIES,
            data=apes,
            group_style={
                'random': {
                    'color': 'black',
                },
                'quantile': {
                    'color': "#A9B0B3",
                    'linestyle': '--',
                }
            },
            y_label="Absolute Relative Error",
            save_as=f"./artefacts/qstrats/{clf}_APE.png",
        )


if __name__ == "__main__":
    main()
