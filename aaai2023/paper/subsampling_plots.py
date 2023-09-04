
from typing import Dict, Tuple, List
from pathlib import Path

from matplotlib import rc_context, pyplot as plt

import numpy as np

from aaai2023.paper.util import read_results, compute_errors, is_subsampling
from aaai2023.paper.names import SUBSAMPLING_SUFFIXES, SUBSAMPLING_PREVALENCES


def boxplots(
    data: Dict[Tuple[str, str], np.array],
    q_strats: List[str],
    suffixes: List[str],
    q_strat_colors: Dict[str, str],
    suffix_map: Dict[str, str],
    y_label: str | None = None,
    save_as: str | None = None,
):
    with rc_context({
        'lines.linewidth': 5,
        'font.size': 20,
    }):
        fig, ax = plt.subplots()
        fig.set_size_inches(16, 9)
        fig.set_tight_layout(True)

        for ix, suff in enumerate(suffixes):
            for jx, qstrat in enumerate(q_strats):
                box_data = plt.boxplot(
                    data[qstrat, suff],
                    positions=[4*ix + jx + 1],
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
                box_color = q_strat_colors[qstrat]
                plt.setp(box_data['boxes'][0], color=box_color)

        ax.set_xticks([4*i + 2 for i in range(len(suffixes))])
        ax.set_xticklabels([suffix_map[s] for s in suffixes])

        if y_label is not None:
            ax.set_ylabel(ylabel=y_label)
        ax.set_xlabel(xlabel="True Prevalence")

        dummy_lines = [
            plt.plot([1, 1], color=q_strat_colors[qstrat], label=qstrat)
            for qstrat in q_strats
        ]
        plt.legend(loc="upper right")

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
        for q_strat in ['CC', "CPCC", "BCC"]:
            for suff in SUBSAMPLING_SUFFIXES:
                # jigsaw has no p10
                if domain == 'jigsaw' and suff == 'p10':
                    continue
                groupings[domain, q_strat, suff] = [
                    r
                    for r in res
                    if r.quant_strategy == q_strat and
                       suff in r.scores_file and
                       domain in r.scores_file
                ]

    aes = {
        k: np.array([e.absolute_error for e in vs])
        for k, vs in groupings.items()
    }
    apes = {
        k: np.array([e.absolute_percentage_error for e in vs])
        for k, vs in groupings.items()
    }
    for domain in subsampling_domains:
        boxplots(
            data={(q, s): vs for (d, q, s), vs in aes.items() if d == domain},
            q_strats=['CC', 'CPCC', 'BCC'],
            suffixes=SUBSAMPLING_SUFFIXES if domain != 'jigsaw' else SUBSAMPLING_SUFFIXES[:-1],
            q_strat_colors={
                "CC": "black",
                "CPCC": "#404749",
                "BCC": "#A9B0B3",
            },
            suffix_map=SUBSAMPLING_PREVALENCES,
            y_label="Absolute Error",
            save_as=f"./artefacts/subsampling/{domain}-all-AE.png",
        )
        boxplots(
            data={(q, s): vs for (d, q, s), vs in apes.items() if d == domain},
            q_strats=['CC', 'CPCC', 'BCC'],
            suffixes=SUBSAMPLING_SUFFIXES if domain != 'jigsaw' else SUBSAMPLING_SUFFIXES[:-1],
            q_strat_colors={
                "CC": "black",
                "CPCC": "#404749",
                "BCC": "#A9B0B3",
            },
            suffix_map=SUBSAMPLING_PREVALENCES,
            y_label="Absolute Error",
            save_as=f"./artefacts/subsampling/{domain}-all-APE.png",
        )

    for clf in ['electra', 'cardiffnlp', 'tfidf-svm', 'perspective']:
        clf_groupings = {
            k: [e for e in vs if clf in e.scores_file]
            for k, vs in groupings.items()
        }
        aes = {
            k: np.array([e.absolute_error for e in vs])
            for k, vs in clf_groupings.items()
        }
        apes = {
            k: np.array([e.absolute_percentage_error for e in vs])
            for k, vs in clf_groupings.items()
        }
        for domain in subsampling_domains:
            boxplots(
                data={(q, s): vs for (d, q, s), vs in aes.items() if d == domain},
                q_strats=['CC', 'CPCC', 'BCC'],
                suffixes=SUBSAMPLING_SUFFIXES if domain != 'jigsaw' else SUBSAMPLING_SUFFIXES[:-1],
                q_strat_colors={
                    "CC": "black",
                    "CPCC": "#404749",
                    "BCC": "#A9B0B3",
                },
                suffix_map=SUBSAMPLING_PREVALENCES,
                y_label="Absolute Error",
                save_as=f"./artefacts/subsampling/{domain}-{clf}-AE.png",
            )
            boxplots(
                data={(q, s): vs for (d, q, s), vs in apes.items() if d == domain},
                q_strats=['CC', 'CPCC', 'BCC'],
                suffixes=SUBSAMPLING_SUFFIXES if domain != 'jigsaw' else SUBSAMPLING_SUFFIXES[:-1],
                q_strat_colors={
                    "CC": "black",
                    "CPCC": "#404749",
                    "BCC": "#A9B0B3",
                },
                suffix_map=SUBSAMPLING_PREVALENCES,
                y_label="Absolute Error",
                save_as=f"./artefacts/subsampling/{domain}-{clf}-APE.png",
            )


if __name__ == '__main__':
    main()
