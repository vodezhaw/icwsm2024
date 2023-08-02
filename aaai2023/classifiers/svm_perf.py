
from pathlib import Path
from typing import Literal, Dict, Optional
import datetime
import subprocess
import hashlib
import json

from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np


BIN_FOLDER = Path("bin")
TEMP_FOLDER = Path("svm_perf_temp")

Loss = Literal["error-rate", "zero-one", "F1", "ROC-AUC", "kld-quant"]

LOSS_ARGS: Dict[Loss, int] = {
    "error-rate": 2,
    "zero-one": 0,
    "F1": 1,
    "ROC-AUC": 10,
    "kld-quant": 99,
}


def svm_perf_format(X, y, fname: Path):
    assert X.shape[0] == len(y)
    assert all(lbl in {0, 1} for lbl in y)

    with fname.open('w') as fout:
        for ix in range(len(y)):
            fout.write("+1" if y[ix] == 1 else "-1")
            for feat_ix in sorted(X[ix].nonzero()[-1]):
                fout.write(f" {feat_ix+1}:{X[ix, feat_ix]}")
            fout.write("\n")


class SVMPerf(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        c: float = .01,
        loss: Loss = "error-rate",
        algorithm: int = 2,
        verbosity: int = 0,
        fit_intercept: bool = True,
        intercept_scaling: float = 1.,
    ):
        self.c = c
        self.loss = loss
        self.algorithm = algorithm
        self.verbosity = max(0, min(3, verbosity))
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling

        self.current_temp: Optional[Path] = None

    def train_file(self) -> Optional[Path]:
        if self.current_temp is None:
            return None
        else:
            return self.current_temp / "train.dat"

    def model_file(self) -> Optional[Path]:
        if self.current_temp is None:
            return None
        else:
            return self.current_temp / "model"

    @staticmethod
    def print_help():
        subprocess.run([BIN_FOLDER / "svm_perf_learn", "-?"])

    def fit(self, X, y):
        self.current_temp = TEMP_FOLDER / datetime.datetime.utcnow().replace(
            microsecond=0).isoformat()
        self.current_temp.mkdir()

        svm_perf_format(X, y, self.train_file())

        subprocess.run([
            BIN_FOLDER / "svm_perf_learn",
            "-v", f"{self.verbosity}",
            "-y", f"{self.verbosity}",
            "-c", f"{self.c}",
            "-l", f"{LOSS_ARGS[self.loss]}",
            "-w", f"{self.algorithm}",
            "--b", f"{self.intercept_scaling if self.fit_intercept else 0}",
            f"{self.train_file()}",
            f"{self.model_file()}",
        ])

        with (self.current_temp / "params.json").open('w') as fout:
            json.dump(obj=self.get_params(), fp=fout, indent=2)

        return self

    def decision_function(self, X, y):
        if self.current_temp is None:
            raise ValueError(f"you need to call `.fit` first")

        data_hash = hashlib.md5(X.data.tobytes()).hexdigest()
        test_file = self.current_temp / f"test_{data_hash}.dat"
        out_file = self.current_temp / f"test_{data_hash}.out"

        if not test_file.exists():
            svm_perf_format(X, y, test_file)
            subprocess.run([
                BIN_FOLDER / "svm_perf_classify",
                "-v", f"{self.verbosity}",
                f"{test_file}",
                f"{self.model_file()}",
                f"{out_file}",
            ])

        n_samples = X.shape[0]
        scores = np.zeros(n_samples)

        with out_file.open('r') as fin:
            for ix, line in enumerate(fin):
                scores[ix] = float(line.strip())

        return scores

    def predict(self, X, y):
        scores = self.decision_function(X, y)
        return scores >= 0.
