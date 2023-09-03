
from pathlib import Path
from typing import Dict

from matplotlib import rc_context
from matplotlib import pyplot as plt

import numpy as np

from aaai2023.paper.util import read_results, compute_errors, is_subsampling
from aaai2023.paper.names import QUANTIFICATION_STRATEGIES


def boxplots(
        data: Dict[str, np.array],
        y_label: str | None = None,
        save_as: str | None = None,
):
    with rc_context({
        "lines.linewidth": 5,
        "font.size": 20,
    }):
        fig, ax = plt.subplots()
        ax.set_ylim(-.05, 1.05)
        fig.set_size_inches(16, 9)
        fig.set_tight_layout(True)

        for ix, qstrat in enumerate(QUANTIFICATION_STRATEGIES):
            _ = plt.boxplot(
                data[qstrat],
                positions=[2*ix + 1],
                widths=.6,
                whis=(5, 95),
                showmeans=True,
                meanline=True,
                boxprops={
                    'linewidth': 5,
                },
                whiskerprops={
                    'linewidth': 2,
                },
                capprops={
                    "linewidth": 2,
                },
                medianprops={
                    'linewidth': 2,
                },
                meanprops={
                    'linewidth': 2,
                }
            )

        ax.set_xticks([2*i + 1 for i in range(len(QUANTIFICATION_STRATEGIES))])
        ax.set_xticklabels(QUANTIFICATION_STRATEGIES)

        if y_label is not None:
            ax.set_ylabel(ylabel=y_label)

        if save_as is None:
            plt.show()
        else:
            plt.savefig(save_as)


def main():
    res_path = Path('./data/results.jsonl')
    res = read_results(res_path)
    res = compute_errors(res)

    # keep only out-of-domain experiments
    res = [
        r
        for r in res
        if r.other_domain_scores_file is not None
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

    res = {
        q_strat: [
            e
            for e in res
            if e.quant_strategy == q_strat
        ]
        for q_strat in QUANTIFICATION_STRATEGIES
    }

    aes = {
        q_strat: np.array([e.absolute_error for e in errs])
        for q_strat, errs in res.items()
    }

    apes = {
        q_strat: np.array([e.absolute_percentage_error for e in errs])
        for q_strat, errs in res.items()
    }

    boxplots(
        aes,
        y_label="Absolute Error",
        save_as="./artefacts/ood/all_aes.png",
    )
    boxplots(
        apes,
        y_label="Absolute Relative Error",
        save_as="./artefacts/ood/all_apes.png",
    )

    for clf in ['electra', 'cardiffnlp', 'tfidf-svm', 'perspective']:
        aes = {
            q_strat: np.array([
                e.absolute_error
                for e in errs
                if Path(e.scores_file).stem.startswith(clf)
            ])
            for q_strat, errs in res.items()
        }
        apes = {
            q_strat: np.array([
                e.absolute_percentage_error
                for e in errs
                if Path(e.scores_file).stem.startswith(clf)
            ])
            for q_strat, errs in res.items()
        }
        boxplots(
            aes,
            y_label="Absolute Error",
            save_as=f"./artefacts/ood/{clf}_aes.png",
        )
        boxplots(
            apes,
            y_label="Absolute Relative Error",
            save_as=f"./artefacts/ood/{clf}_apes.png",
        )


if __name__ == '__main__':
    main()
