
from pathlib import Path

from aaai2023.paper.util import read_results, is_subsampling, compute_errors
from aaai2023.paper.names import QUANTIFICATION_STRATEGIES, SELECTION_STRATEGIES
from aaai2023.paper.plot_utils import render_quant_results


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

    data = {}
    rows = []
    for q_strat in ["PACC", "CPCC", "BCC"]:
        for n in sample_nums:
            row_name = f"{q_strat}-{n}"
            rows.append(row_name)
            for s_strat in SELECTION_STRATEGIES:
                data[row_name, s_strat.title()] = [
                    e
                    for e in res
                    if (
                        e.quant_strategy == q_strat
                        and e.n_samples_to_select == n
                        and e.sample_selection_strategy == s_strat
                    )
                ]

    for error_type in ["AE"]:
        render_quant_results(
            rows=rows,
            columns=[s.title() for s in SELECTION_STRATEGIES],
            data=data,
            error_type=error_type,
            save_as=f"paper_plots/fewer_samples/selected_{error_type}.tex",
        )



if __name__ == "__main__":
    main()
