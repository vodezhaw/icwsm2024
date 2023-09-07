
from pathlib import Path

from aaai2023.paper.util import read_results, compute_errors, is_subsampling
from aaai2023.paper.names import QUANTIFICATION_STRATEGIES
from aaai2023.paper.plot_utils import all_box_plots


def main():
    res_path = Path('./data/results/out_of_domain.jsonl')
    res = read_results(res_path)
    res = compute_errors(res)

    # keep only out-of-domain experiments
    assert all(r.other_domain_scores_file is not None for r in res)

    # ignore sub-sampling experiments
    assert all(not is_subsampling(Path(r.scores_file)) for r in res)

    # ignore errors
    assert all(r.error_message is None for r in res)

    groupings = {}
    for q_strat in QUANTIFICATION_STRATEGIES:
        groupings[q_strat, "dummy"] = [
            e
            for e in res
            if e.quant_strategy == q_strat
        ]

    all_box_plots(
        groupings=groupings,
        x_labels=QUANTIFICATION_STRATEGIES,
        group_labels=['dummy'],
        group_style={
            'dummy': {
                'color': 'black',
            }
        },
        file_name_fn=lambda clf, err: f"./paper_plots/ood/{clf}_{err}.png",
    )


if __name__ == '__main__':
    main()
