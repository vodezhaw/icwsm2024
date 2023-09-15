
from pathlib import Path

from icwsm2024.paper.util import read_results, is_subsampling, compute_errors
from icwsm2024.paper.plot_utils import render_quant_results, simple_boxplot
from icwsm2024.paper.names import QUANTIFICATION_STRATEGIES, SELECTION_STRATEGIES


def main():
    res_path = Path('./data/results/compare_quantification_strategies.jsonl')
    res = read_results(res_path)
    res = compute_errors(res)

    # keep only in-domain experiments
    assert all(r.other_domain_scores_file is None for r in res)

    # ignore sub-sampling experiments
    assert all(not is_subsampling(Path(r.scores_file)) for r in res)

    # ignore errors
    assert all(r.error_message is None for r in res)

    groupings = {}
    for q_strat in QUANTIFICATION_STRATEGIES:
        for s_strat in SELECTION_STRATEGIES:
            groupings[q_strat, s_strat.title()] = [
                e
                for e in res
                if e.sample_selection_strategy == s_strat and e.quant_strategy == q_strat
            ]

    for error_type in ["AE", "SAPE"]:
        render_quant_results(
            rows=QUANTIFICATION_STRATEGIES,
            columns=[s.title() for s in SELECTION_STRATEGIES],
            data=groupings,
            error_type=error_type,
            save_as=f"paper_plots/compare_quantification_strategies/all_{error_type}.tex",
        )
        simple_boxplot(
            x_labels=QUANTIFICATION_STRATEGIES,
            data={
                q_strat: groupings[q_strat, 'Random']
                for q_strat in QUANTIFICATION_STRATEGIES
            },
            error_type=error_type,
            x_label=None,
            y_label="Absolute Error (AE)",
            save_as=f"paper_plots/compare_quantification_strategies/random_{error_type}.png",
        )


if __name__ == "__main__":
    main()
