
from pathlib import Path
from typing import Dict

import numpy as np
from matplotlib import rc_context
from matplotlib import pyplot as plt

from aaai2023.paper.util import read_results, is_subsampling, compute_errors


def boxplots(
    data: Dict[str, Dict[str, np.array]],
    y_label: str | None = None,
    save_as: str | None = None,
):
    with rc_context({
        "lines.linewidth": 5,
        "font.size": 20,
        # "font.weight": "bold",
        # "axes.labelweight": "bold",
    }):
        fig, ax = plt.subplots()
        fig.set_size_inches(16, 9)
        fig.set_tight_layout(True)

        qstrats = ['CC', 'ACC', 'PCC', 'PACC', 'CPCC', 'BCC']
        sstrats = [('random', "black"), ('quantile', "#A9B0B3")]

        for ix, qstrat in enumerate(qstrats):
            for jx, (sstrat, box_color) in enumerate(sstrats):
                box_data = plt.boxplot(
                    data[qstrat][sstrat],
                    positions=[3*ix + jx + 1],
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
                plt.setp(box_data['boxes'][0], color=box_color)

        ax.set_xticks([3*i + 1.5 for i in range(len(qstrats))])
        ax.set_xticklabels(qstrats)

        if y_label is not None:
            ax.set_ylabel(ylabel=y_label)

        dummy_lines = [
            plt.plot([1, 1], color=color, label=sstrat)
            for sstrat, color in sstrats
        ]
        plt.legend()

        for dummy in dummy_lines:
            dummy[0].set_visible(False)

        if save_as is None:
            plt.show()
        else:
            plt.savefig(save_as)


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

    res = {
        q_strat: {
            s_strat: [
                e
                for e in res
                if e.sample_selection_strategy == s_strat and e.quant_strategy == q_strat
            ]
            for s_strat in ['random', 'quantile']
        }
        for q_strat in ['CC', 'ACC', 'PCC', 'PACC', 'CPCC', 'BCC']
    }

    aes = {
        q_strat: {
            s_strat: np.array([e.absolute_error for e in errs])
            for s_strat, errs in s_strats.items()
        }
        for q_strat, s_strats in res.items()
    }

    apes = {
        q_strat: {
            s_strat: np.array([e.absolute_percentage_error for e in errs])
            for s_strat, errs in s_strats.items()
        }
        for q_strat, s_strats in res.items()
    }

    boxplots(
        aes,
        y_label="Absolute Error",
        save_as="./artefacts/all_aes.png",
    )
    boxplots(
        apes,
        y_label="Absolute Relative Error",
        save_as="./artefacts/all_apes.png",
    )

    for clf in ['electra', 'cardiffnlp', 'tfidf-svm', 'perspective']:
        aes = {
            q_strat: {
                s_strat: np.array([
                    e.absolute_error
                    for e in errs
                    if Path(e.scores_file).stem.startswith(clf)
                ])
                for s_strat, errs in s_strats.items()
            }
            for q_strat, s_strats in res.items()
        }
        apes = {
            q_strat: {
                s_strat: np.array([
                    e.absolute_percentage_error
                    for e in errs
                    if Path(e.scores_file).stem.startswith(clf)
                ])
                for s_strat, errs in s_strats.items()
            }
            for q_strat, s_strats in res.items()
        }
        boxplots(
            aes,
            y_label="Absolute Error",
            save_as=f"./artefacts/{clf}_aes.png",
        )
        boxplots(
            apes,
            y_label="Absolute Relative Error",
            save_as=f"./artefacts/{clf}_apes.png",
        )

    return res


if __name__ == "__main__":
    main()
