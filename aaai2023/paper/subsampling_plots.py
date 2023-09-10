
from pathlib import Path

from aaai2023.paper.util import read_results, compute_errors, is_subsampling
from aaai2023.paper.names import SUBSAMPLING_PREVALENCES
from aaai2023.paper.plot_utils import all_box_plots


def main():
    res_path = Path('./data/results/prevalence_subsampling.jsonl')
    res = read_results(res_path)
    res = compute_errors(res)

    subsampling_domains = [
        "jigsaw",
        "ex-machina",
    ]

    # keep only in-domain experiments
    assert all(r.other_domain_scores_file is None for r in res)

    # keep only subsampling experiments
    assert all(is_subsampling(Path(r.scores_file)) for r in res)

    # ignore errors
    res = [
        r
        for r in res
        if r.error_message is None
    ]
    
    suffixes_to_plot = ["p2", "p3", "p5", "p7", "p10"]

    groupings = {}
    for domain in subsampling_domains:
        for suff in suffixes_to_plot:
            for q_strat in ['CC', "PACC", "CPCC", "BCC"]:
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
                    suffixes_to_plot[:-1] if domain == 'jigsaw' else suffixes_to_plot
                )
            ],
            group_labels=['CC', "PACC", 'CPCC', 'BCC'],
            group_style={
                'CC': {
                    'color': 'black',
                },
                "PACC": {
                    "color": "#919999",
                    "linestyle": "-."
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
            file_name_fn=lambda clf, err: f"./paper_plots/subsampling/{domain}-{clf}-{err}.png",
            x_label="Prevalence",
        )

    pooled = {}
    for (d, prev, q), vs in groupings.items():
        if pooled.get((prev, q)) is None:
            pooled[prev, q] = []
        pooled[prev, q] += vs

    all_box_plots(
        groupings=pooled,
        x_labels=[
            f"{SUBSAMPLING_PREVALENCES[suff]:.3f}"
            for suff in suffixes_to_plot
        ],
        group_labels=['CC', "PACC", 'CPCC', 'BCC'],
        group_style={
            'CC': {
                'color': 'black',
            },
            "PACC": {
                "color": "#919999",
                "linestyle": "-."
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
        file_name_fn=lambda clf, err: f"./paper_plots/subsampling/both-{clf}-{err}.png",
        x_label="Prevalence",
    )




if __name__ == '__main__':
    main()
