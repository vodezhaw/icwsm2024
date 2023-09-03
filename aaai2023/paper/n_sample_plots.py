
from pathlib import Path
from typing import Dict, Tuple, List

from matplotlib import rc_context
from matplotlib import pyplot as plt

import numpy as np

from aaai2023.paper.util import read_results, is_subsampling, compute_errors
from aaai2023.paper.names import QUANTIFICATION_STRATEGIES, SELECTION_STRATEGIES


def boxplots(
    data: Dict[Tuple[str, int], np.array],
    selection_strats: List[str],
    sample_nums: List[int],
    selection_strat_colors: Dict[str, str],
    y_label: str | None = None,
    save_as: str | None = None,
):
    with rc_context({
        "lines.linewidth": 5,
        "font.size": 20,
    }):
        fig, ax = plt.subplots()
        fig.set_size_inches(16, 9)
        fig.set_tight_layout(True)

        for ix, n in enumerate(sample_nums):
            for jx, sstrat in enumerate(selection_strats):
                box_color = selection_strat_colors[sstrat]
                box_data = plt.boxplot(
                    data[sstrat, n],
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

        ax.set_xticks([3*i + 1.5 for i in range(len(sample_nums))])
        ax.set_xticklabels(sample_nums)

        if y_label is not None:
            ax.set_ylabel(ylabel=y_label)

        dummy_lines = [
            plt.plot([1, 1], color=selection_strat_colors[sstrat], label=sstrat)
            for sstrat in selection_strats
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

    sample_nums = sorted({
        e.n_samples_to_select
        for e in res
    })

    groupings = {}
    for qstrat in QUANTIFICATION_STRATEGIES:
        for sstrat in SELECTION_STRATEGIES:
            for n in sample_nums:
                groupings[qstrat, sstrat, n] = [
                    e
                    for e in res
                    if e.quant_strategy == qstrat and
                       e.sample_selection_strategy == sstrat and
                       e.n_samples_to_select == n
                ]

    all_aes = {
        k: np.array([e.absolute_error for e in errs])
        for k, errs in groupings.items()
    }
    all_apes = {
        k: np.array([e.absolute_percentage_error for e in errs])
        for k, errs in groupings.items()
    }
    
    for qstrat in QUANTIFICATION_STRATEGIES:
        aes = {
            (s, n): ae
            for (q, s, n), ae in all_aes.items()
            if q == qstrat
        }
        boxplots(
            data=aes,
            selection_strats=SELECTION_STRATEGIES,
            sample_nums=sample_nums,
            selection_strat_colors={
                "random": "black",
                "quantile": "#A9B0B3",
            },
            y_label="Absolute Error",
            save_as=f"./artefacts/{qstrat}_sample_nums_AE.png",
        )
        
        apes = {
            (s, n): ae
            for (q, s, n), ae in all_apes.items()
            if q == qstrat
        }
        boxplots(
            data=apes,
            selection_strats=SELECTION_STRATEGIES,
            sample_nums=sample_nums,
            selection_strat_colors={
                "random": "black",
                "quantile": "#A9B0B3",
            },
            y_label="Absolute Percentage Error",
            save_as=f"./artefacts/{qstrat}_sample_nums_APE.png",
        )


if __name__ == "__main__":
    main()
