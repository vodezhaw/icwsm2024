
from pathlib import Path

from aaai2023.paper.util import read_results, compute_errors, is_subsampling
from aaai2023.paper.names import SUBSAMPLING_PREVALENCES
from aaai2023.paper.plot_utils import render_quant_results


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

    table_data = {}
    for q_strat in ['CC', 'PACC', 'CPCC', 'BCC']:
        for suff in suffixes_to_plot:
            table_data[f"{SUBSAMPLING_PREVALENCES[suff]:.3f}", q_strat] = [
                r
                for d in subsampling_domains
                for r in groupings.get((d, f"{SUBSAMPLING_PREVALENCES[suff]:.3f}", q_strat), [])
            ]

    for error_type in ["AE"]:
        render_quant_results(
            rows=[f"{SUBSAMPLING_PREVALENCES[suff]:.3f}" for suff in suffixes_to_plot],
            columns=['CC', 'PACC', 'CPCC', 'BCC'],
            data=table_data,
            error_type=error_type,
            save_as=f"paper_plots/low_prevalence/selected_{error_type}.tex",
        )


if __name__ == '__main__':
    main()
