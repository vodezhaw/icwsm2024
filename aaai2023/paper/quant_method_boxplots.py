
from pathlib import Path

from aaai2023.paper.util import read_results, is_subsampling, compute_errors
from aaai2023.paper.plot_utils import all_box_plots
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

    all_box_plots(
        groupings=groupings,
        x_labels=QUANTIFICATION_STRATEGIES,
        group_labels=SELECTION_STRATEGIES,
        group_style={
            'random': {
                'color': 'black',
            },
            'quantile': {
                'color': "#A9B0B3",
                'linestyle': '--',
            }
        },
        file_name_fn=lambda clf, err: f"./artefacts/qstrats/{clf}_{err}.png",
    )


if __name__ == "__main__":
    main()
