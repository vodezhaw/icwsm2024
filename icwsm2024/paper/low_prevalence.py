
from pathlib import Path

from icwsm2024.paper.util import read_results, compute_errors, is_subsampling
from icwsm2024.paper.names import SUBSAMPLING_PREVALENCES, SELECTION_STRATEGIES
from icwsm2024.paper.plot_utils import render_quant_results


def main():
    res_path = Path('./data/results/prevalence_subsampling.jsonl')
    res = read_results(res_path)
    res = compute_errors(res)

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

    for s_strat in SELECTION_STRATEGIES:
        groupings = {}
        for suff in suffixes_to_plot:
            for q_strat in ['CC', "PACC", "CPCC", "BCC"]:
                groupings[f"{SUBSAMPLING_PREVALENCES[suff]:.3f}", q_strat] = [
                    r
                    for r in res
                    if r.quant_strategy == q_strat and suff in r.scores_file and r.sample_selection_strategy == s_strat
                ]

        for error_type in ["AE", "SAPE"]:
            render_quant_results(
                rows=[f"{SUBSAMPLING_PREVALENCES[suff]:.3f}" for suff in suffixes_to_plot],
                columns=['CC', 'PACC', 'CPCC', 'BCC'],
                data=groupings,
                error_type=error_type,
                save_as=f"paper_plots/low_prevalence/selected_{s_strat}_{error_type}.tex",
            )


if __name__ == '__main__':
    main()
