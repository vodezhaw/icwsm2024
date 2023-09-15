
from dataclasses import dataclass


import numpyro
import numpyro.distributions as dist


from icwsm2024.models import Binomial


@dataclass(frozen=True)
class BCCPriorData:
    annotations: Binomial
    tpr_data: Binomial
    fpr_data: Binomial
    clf_data: Binomial


def bcc_model(bcc_prior: BCCPriorData):
    p_true = numpyro.sample(
        "p_true",
        dist.Beta(bcc_prior.annotations.pos() + 1, bcc_prior.annotations.neg() + 1),
    )
    tpr = numpyro.sample(
        "tpr",
        dist.Beta(bcc_prior.tpr_data.pos() + 1, bcc_prior.tpr_data.neg() + 1),
    )
    fpr = numpyro.sample(
        "fpr",
        dist.Beta(bcc_prior.fpr_data.pos() + 1, bcc_prior.fpr_data.neg() + 1),
    )
    p_obs = numpyro.deterministic(
        "p_obs",
        p_true*tpr + (1. - p_true)*fpr,
    )
    _ = numpyro.sample(
        "obs",
        dist.Binomial(total_count=bcc_prior.clf_data.n(), probs=p_obs),
        obs=bcc_prior.clf_data.pos(),
    )
