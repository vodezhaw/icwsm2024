
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from aaai2023.datasets.classifier import TestDataset, ScoredDataset
from aaai2023.classifiers.svm_perf import SVMPerf
from aaai2023.models import Binomial
from aaai2023.models.bcc import bcc_model, BCCPriorData
from aaai2023.models.mcmc import run_mcmc


class ThresholdMethod:

    @abstractmethod
    def threshold_quality(self, tpr: np.array, fpr: np.array) -> np.array:
        raise NotImplemented


class ThresholdAt50(ThresholdMethod):

    def threshold_quality(self, tpr: np.array, fpr: np.array) -> np.array:
        # argmax has tpr value closest to .5
        return -np.abs(tpr - .5)


class MethodX(ThresholdMethod):

    def threshold_quality(self, tpr: np.array, fpr: np.array) -> np.array:
        # argmax where almost fpr = 1. - tpr
        return -(fpr - (1. - tpr))


class MethodMax(ThresholdMethod):

    def threshold_quality(self, tpr: np.array, fpr: np.array) -> np.array:
        return tpr - fpr


@dataclass(frozen=True)
class BinaryClassifierData:
    labels: np.array
    scores: np.array
    default_threshold: float = .5

    @staticmethod
    def from_test_scores(
        test_set: TestDataset,
        scores: ScoredDataset,
    ) -> 'BinaryClassifierData':
        if not scores.test_data.startswith(test_set.name):
            raise ValueError(f"trying to combine incompatible "
                             f"TestDataset ({test_set.name}) and"
                             f" ScoredDataset ({scores.classifier_name, scores.train_data, scores.test_data})")

        score_map = {
            s.id: s.score
            for s in scores.scores
        }
        label_map = {
            s.id: s.label
            for s in test_set.test_samples
        }
        id_seq = [s.id for s in test_set.test_samples if s.id in score_map.keys()]
        labels = np.array([label_map[i] for i in id_seq], dtype=int)
        ss = np.array([score_map[i] for i in id_seq])

        return BinaryClassifierData(
            labels=labels,
            scores=ss,
            default_threshold=scores.default_threshold,
        )

    def __post_init__(self):
        self.__check_lengths()

    def __check_lengths(self):
        if self.labels.shape != self.scores.shape:
            raise ValueError(f"labels and scores dont match")

    def __len__(self):
        return len(self.labels)

    def merge(self, other: 'BinaryClassifierData') -> 'BinaryClassifierData':
        assert self.default_threshold == other.default_threshold
        return BinaryClassifierData(
            labels=np.concatenate([self.labels, other.labels]),
            scores=np.concatenate([self.scores, other.scores]),
            default_threshold=self.default_threshold,
        )

    def prevalence(self) -> float:
        return float(self.labels.mean())

    def threshold(self, method: ThresholdMethod | None = None) -> float:
        if method is None:
            return self.default_threshold
        tpr, fpr, th = roc_curve(
            y_true=self.labels,
            y_score=self.scores,
            pos_label=1,
            drop_intermediate=False,
        )
        quality = method.threshold_quality(tpr, fpr)
        return th[quality.argmax()]

    def clf_labels(self, threshold: Optional[float] = None) -> np.array:
        if threshold is None:
            threshold = self.default_threshold
        y_pred = (self.scores >= threshold).astype(int)
        return y_pred

    def tpr_data(self, threshold: Optional[float] = None) -> Binomial:
        y_pred = self.clf_labels(threshold)
        return Binomial(
            n_pos_=y_pred[self.labels == 1].sum(),
            total_=(self.labels == 1).sum(),
        )

    def tpr(self, threshold: Optional[float] = None) -> float:
        return self.tpr_data(threshold).p()

    def expected_tpr(self) -> float:
        return self.scores[self.labels == 1].mean()

    def fpr_data(self, threshold: Optional[float] = None) -> Binomial:
        y_pred = self.clf_labels(threshold)
        return Binomial(
            n_pos_=y_pred[self.labels == 0].sum(),
            total_=(self.labels == 0).sum(),
        )

    def fpr(self, threshold: Optional[float] = None) -> float:
        return self.fpr_data(threshold).p()

    def expected_fpr(self) -> float:
        return self.scores[self.labels == 0].mean()

    def clf_counts(self, threshold: Optional[float]) -> Binomial:
        y_pred = self.clf_labels(threshold)
        return Binomial(
            n_pos_=y_pred.sum(),
            total_=len(y_pred),
        )

    def classify_and_count(self, threshold: Optional[float] = None) -> float:
        return float(self.clf_counts(threshold).p())

    def probabilistic_classify_and_count(self) -> float:
        return float(self.scores.mean())

    def adjusted_classify_and_count(
        self,
        tpr: float,
        fpr: float,
        threshold: Optional[float] = None,
    ) -> float:
        cc = self.classify_and_count(threshold)
        acc = (cc - fpr) / (tpr - fpr)
        return min(1., max(acc, 0.))

    def probabilistic_adjusted_classify_and_count(
        self,
        expected_tpr: float,
        expected_fpr: float,
    ) -> float:
        pcc = self.probabilistic_classify_and_count()
        pacc = (pcc - expected_fpr) / (expected_tpr - expected_fpr)
        return min(1., max(pacc, 0.))

    def random_split(
        self,
        n_dev: int,
        random_state: int = 0xdeadbeef,
        n_quantiles: Optional[int] = None,
    ) -> 'BinaryQuantificationData':

        if n_quantiles is not None:
            quantile_labels = np.ones(len(self)).astype(int) * n_quantiles
            for i in range(1, n_quantiles):
                q = np.quantile(self.scores, i / n_quantiles)
                quantile_labels[self.scores <= q] -= 1
            stratify = quantile_labels
        else:
            stratify = None

        dev_ixs, test_ixs = train_test_split(
            np.arange(len(self)),
            train_size=n_dev,
            random_state=random_state,
            shuffle=True,
            stratify=stratify,
        )
        return BinaryQuantificationData(
            dev=BinaryClassifierData(
                labels=self.labels[dev_ixs],
                scores=self.scores[dev_ixs],
                default_threshold=self.default_threshold,
            ),
            test=BinaryClassifierData(
                labels=self.labels[test_ixs],
                scores=self.scores[test_ixs],
                default_threshold=self.default_threshold,
            )
        )

    def subsample(
        self,
        n: int,
        p: float | None = None,
        random_state: int = 0xdeadbeef,
    ) -> 'BinaryClassifierData':
        if p is None:
            keep_ixs, _ = train_test_split(
                np.arange(len(self)),
                train_size=n,
                random_state=random_state,
                shuffle=True,
                stratify=self.labels,
            )
            return BinaryClassifierData(
                scores=self.scores[keep_ixs],
                labels=self.labels[keep_ixs],
                default_threshold=self.default_threshold,
            )
        else:
            n_pos = int((p * n) + .5)
            n_neg = n - n_pos

            all_ixs = np.arange(len(self))
            pos_mask = self.labels == 1

            pos_keep, _ = train_test_split(
                all_ixs[pos_mask],
                train_size=n_pos,
                random_state=random_state,
                shuffle=True,
                stratify=None,
            )
            neg_keep, _ = train_test_split(
                all_ixs[~pos_mask],
                train_size=n_neg,
                random_state=random_state,
                shuffle=True,
                stratify=None,
            )

            keep = np.zeros(len(self)).astype(bool)
            keep[pos_keep] = True
            keep[neg_keep] = True

            return BinaryClassifierData(
                scores=self.scores[keep],
                labels=self.labels[keep],
                default_threshold=self.default_threshold,
            )


def merge_many(*args) -> 'BinaryClassifierData':
    assert len({a.default_threshold for a in args}) == 1
    return BinaryClassifierData(
        scores=np.concatenate([a.scores for a in args]),
        labels=np.concatenate([a.labels for a in args]),
        default_threshold=args[0].default_threshold,
    )


class PlattScaling:

    def __init__(self):
        self.logreg = LogisticRegression(
            C=1.,
            fit_intercept=True,
            penalty=None,
            class_weight=None,
            random_state=0xdeadbeef,
        )

    def fit(self, y_scores, y_true):
        self.logreg.fit(y_scores[:, np.newaxis], y_true)
        return self

    def transform(self, y_scores):
        probas = self.logreg.predict_proba(y_scores[:, np.newaxis])
        return probas[:, 1]


class QuantScaling:

    def __init__(self):
        self.svm_perf = SVMPerf(
            c=.01,
            loss="kld-quant",
            verbosity=0,
        )

    def fit(self, y_scores, y_true):
        self.svm_perf.fit(y_scores[:, np.newaxis], y_true)
        return self

    def transform(self, y_scores, y_true):
        return self.svm_perf.decision_function(y_scores[:, np.newaxis], y_true)


@dataclass(frozen=True)
class BinaryQuantificationData:
    dev: BinaryClassifierData
    test: BinaryClassifierData

    def true_prevalence(self):
        return float(self.test.labels.mean())

    def classify_and_count(self, method: ThresholdMethod | None = None) -> float:
        th = self.dev.threshold(method)
        return self.test.classify_and_count(th)

    def probabilistic_classify_and_count(
        self,
    ) -> float:
        return self.test.probabilistic_classify_and_count()

    def quant_cc(self) -> float:
        quant = QuantScaling()
        quant.fit(self.dev.scores, self.dev.labels)
        return float((quant.transform(self.test.scores, self.test.labels) >= 0.).mean())

    def calibrated_pcc(
        self,
    ) -> float:
        calib = PlattScaling()
        calib.fit(self.dev.scores, self.dev.labels)
        return float(calib.transform(self.test.scores).mean())

    def adjusted_classify_and_count(
        self,
        method: ThresholdMethod | None = None,
    ) -> float:
        th = self.dev.threshold(method)
        tpr = self.dev.tpr(th)
        fpr = self.dev.fpr(th)
        return self.test.adjusted_classify_and_count(tpr=tpr, fpr=fpr, threshold=th)

    def probabilistic_adjusted_classify_and_count(self) -> float:
        tpr = self.dev.expected_tpr()
        fpr = self.dev.expected_fpr()
        return self.test.probabilistic_adjusted_classify_and_count(
            expected_tpr=tpr, expected_fpr=fpr)

    def bayesian_classify_and_count(
        self,
        method: ThresholdMethod | None = None,
        agnostic: bool = True,
    ) -> float:
        th = self.dev.threshold(method)

        bcc_prior = BCCPriorData(
            annotations=Binomial(0, 0) if agnostic else Binomial(self.dev.labels.sum(), len(self.dev)),
            tpr_data=self.dev.tpr_data(th),
            fpr_data=self.dev.fpr_data(th),
            clf_data=self.test.clf_counts(th),
        )

        samples = run_mcmc(bcc_model, bcc_prior, verbose=False)

        return float(samples['p_true'].mean())
