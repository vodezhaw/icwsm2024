
from pathlib import Path

from icwsm2024.paper.util import read_results, is_subsampling, compute_errors
from icwsm2024.paper.names import SELECTION_STRATEGIES
from icwsm2024.paper.plot_utils import render_quant_results, simple_boxplot


def main():
    res_path = Path('./data/results/calibration_methods.jsonl')
    res = read_results(res_path)
    res = compute_errors(res)

    assert all(r.other_domain_scores_file is None for r in res)
    assert all(not is_subsampling(Path(r.scores_file)) for r in res)
    assert all(r.error_message is None for r in res)

    q_strats = [
        "CPCC",
        "CPCC-ISO",
        "CPCC-HB10",
        "CPCC-HB100",
    ]

    groupings = {}
    for q_strat in q_strats:
        for s_strat in SELECTION_STRATEGIES:
            groupings[q_strat, s_strat.title()] = [
                e
                for e in res
                if e.sample_selection_strategy == s_strat and e.quant_strategy == q_strat
            ]

    for error_type in ["AE", "SAPE"]:
        render_quant_results(
           rows=q_strats,
           columns=[s.title() for s in SELECTION_STRATEGIES],
           data=groupings,
           error_type=error_type,
           save_as=f"paper_plots/calibration_methods/all_{error_type}.tex",
        )
        simple_boxplot(
            x_labels=q_strats,
            data={
                q_strat: groupings[q_strat, 'Random']
                for q_strat in q_strats
            },
            error_type=error_type,
            x_label=None,
            y_label=error_type,
            save_as=f"paper_plots/calibration_methods/random_{error_type}.png",
        )


if __name__ == "__main__":
    main()
