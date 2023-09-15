
import json
from pathlib import Path
from dataclasses import asdict

from icwsm2024.paper.experiment_configurations import (
    compare_quantification_strategies,
    out_of_domain,
    prevalence_subsampling,
    sample_sizes,
)


def main(
    scores_folder: Path,
    experiment_mode: str,
):
    if experiment_mode == "compare_quantification_strategies":
        gen = compare_quantification_strategies(scores_folder=scores_folder)
    elif experiment_mode == "out_of_domain":
        gen = out_of_domain(scores_folder=scores_folder)
    elif experiment_mode == "prevalence_subsampling":
        gen = prevalence_subsampling(scores_folder=scores_folder)
    elif experiment_mode == "sample_sizes":
        gen = sample_sizes(scores_folder=scores_folder)
    else:
        raise ValueError(f"unknown experiment mode `{experiment_mode}`")

    for e in gen:
        res_dict = asdict(e)
        res_dict['hash_id'] = e.compute_db_hash()
        print(json.dumps(res_dict))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", '--scores', dest="scores", required=True, type=Path)
    parser.add_argument(
        "-m", "--mode", dest="mode", required=True, type=str)
    args = parser.parse_args()

    main(
        scores_folder=args.scores,
        experiment_mode=args.mode,
    )
