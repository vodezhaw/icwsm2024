
from typing import Optional
import json
from pathlib import Path

from aaai2023.classifiers.scripts.finetune_full import main as finetune_hf
from aaai2023.classifiers.scripts.run_perspective import main as run_perspective


def main(config: Optional[dict] = None):
    if config is None:
        raise ValueError(f"no configuration provided")

    data_path = Path(config["data"])
    output_path = Path(config['output'])
    scores_path = Path(config['scores'])
    mode = config.get('mode', "")
    is_dev = config.get('dev', False)

    sub_config = config[mode]

    if mode == "finetune_huggingface":
        finetune_hf(
            train_path=data_path / "train",
            test_path=data_path / "test",
            scores_path=scores_path,
            model_out=output_path / sub_config['short_name'],
            model_short=sub_config['short_name'],
            hf_model=sub_config['base_model'],
            dev_mode=is_dev,
        )
    elif mode == "run_perspective":
        run_perspective(
            test_path=data_path / "test",
            scores_path=scores_path,
            api_key=sub_config['api_key'],
            cache_file=Path(sub_config['cache']),
        )
    else:
        print(f"unknown mode '{mode}' in configuration:"
              f"\ndont know what to run")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", "-f",
        dest="config_file",
        required=False,
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--config", "-c",
        dest="config_str",
        required=False,
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if args.config_file is not None:
        with args.config_file.open('r') as fin:
            config = json.load(fin)
    elif args.config_str is not None:
        config = json.loads(args.config_str)
    else:
        config = None

    main(config=config)
