
from pathlib import Path

from aaai2023.paper.util import read_results, is_subsampling, compute_errors
from aaai2023.paper.names import QUANTIFICATION_STRATEGIES, SELECTION_STRATEGIES
from aaai2023.paper.plot_utils import all_box_plots


def main():
    res_path = Path('./data/results/sample_sizes.jsonl')
    res = read_results(res_path)
    res = compute_errors(res)

    # keep only in-domain experiments
    assert all(r.other_domain_scores_file is None for r in res)

    # ignore sub-sampling experiments
    assert all(not is_subsampling(Path(r.scores_file)) for r in res)

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

    for qstrat in QUANTIFICATION_STRATEGIES:
        groups = {
            (n, s): errs
            for (q, s, n), errs in groupings.items()
            if q == qstrat
        }
        all_box_plots(
            groupings=groups,
            x_labels=sample_nums,
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
            file_name_fn=lambda clf, err: f"./paper_plots/sample_sizes/{qstrat}_{clf}_{err}.png",
            x_label="N Selected",
        )


if __name__ == "__main__":
    main()
