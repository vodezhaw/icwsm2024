
from pathlib import Path
import json
from dataclasses import asdict

import numpyro

from aaai2023.paper.experiments import ExperimentWrapper
from aaai2023.paper.util import Experiment


def main(
    test_folder: Path,
    experiment_json: str,
):
    numpyro.set_host_device_count(1)

    wrapper = ExperimentWrapper(test_folder=str(test_folder))

    e_dict = json.loads(experiment_json)
    db_id = e_dict['hash_id']
    del e_dict['hash_id']

    e = Experiment(**e_dict)

    res = wrapper(e)

    res_dict = asdict(res)
    res_dict['hash_id'] = db_id

    print(json.dumps(res_dict))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--test", dest="test", required=True, type=Path)
    parser.add_argument(
        "-e", "--experiment", dest="experiment", required=True, type=str)
    args = parser.parse_args()

    main(
        test_folder=args.test,
        experiment_json=args.experiment,
    )
