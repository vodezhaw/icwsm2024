
import time
import shelve
import json
import gzip
from pathlib import Path

from tqdm import tqdm

from aaai2023.classifiers.perspective import Perspective
from aaai2023.datasets.classifier import (
    TestDataset,
    ScoredDataset,
    Sample,
    ScoredSample,
)


class CachedApi:

    def __init__(self, api_key: str, cache_file: Path):
        self.api = Perspective(api_key=api_key)
        self.cache_file = cache_file

    def __enter__(self):
        self.db = shelve.open(str(self.cache_file))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()

    def __call__(self, sample: Sample):
        res = self.db.get(sample.id)
        if res is None:
            res = self.api(text=sample.text)
            time.sleep(1.01)
            self.db[sample.id] = res
        return res


def main(
    test_path: Path,
    scores_path: Path,
    api_key: str,
    cache_file: Path,
):
    for test_f in tqdm(test_path.glob("*.json.gz")):
        with gzip.open(test_f, 'rt') as fin:
            test_data = TestDataset.from_json(json.load(fin))

        score_file = scores_path / f"perspective-{test_data.name}.json.gz"
        if score_file.exists():
            continue

        print(f"classifying {test_data.name}")

        with CachedApi(api_key=api_key, cache_file=cache_file) as api:
            scored_samples = []
            for s in tqdm(test_data.test_samples):
                score_dict = api(s)

                score = max(
                    entry['summaryScore']['value']
                    for entry in score_dict['attributeScores'].values()
                )
                scored_samples.append(ScoredSample(id=s.id, score=score))

            clf_params = {
                "languages": api.api.languages,
                "requestedAttributes": api.api.attributes,
                "url": api.api.url,
            }

        out_data = ScoredDataset(
            classifier_name="perspective-api",
            train_data="online-api",
            test_data=test_data.name,
            classifier_params=clf_params,
            default_threshold=.5,
            scores=scored_samples,
        )

        with gzip.open(score_file, 'wt') as fout:
            fout.write(json.dumps(obj=out_data.json(), indent=2))
            fout.write('\n')