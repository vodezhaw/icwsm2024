
from pathlib import Path

from icwsm2024.paper.util import read_results, compute_errors, is_subsampling
from icwsm2024.paper.names import QUANTIFICATION_STRATEGIES
from icwsm2024.paper.plot_utils import render_quant_results, simple_boxplot


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
        groupings[q_strat, "Out-of-Domain"] = [
            e
            for e in res
            if e.quant_strategy == q_strat
        ]

    for error_type in ["AE", "SAPE"]:
        simple_boxplot(
            x_labels=QUANTIFICATION_STRATEGIES,
            data={
                q_strat: groupings[q_strat, "Out-of-Domain"]
                for q_strat in QUANTIFICATION_STRATEGIES
            },
            error_type=error_type,
            x_label=None,
            y_label="Absolute Error (AE)",
            save_as=f"paper_plots/out_of_domain/all_{error_type}.png",
        )
        render_quant_results(
            rows=QUANTIFICATION_STRATEGIES,
            columns=["Out-of-Domain"],
            data=groupings,
            error_type=error_type,
            save_as=f"paper_plots/out_of_domain/all_{error_type}.tex",
        )


if __name__ == '__main__':
    main()
