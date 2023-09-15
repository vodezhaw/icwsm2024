
import jax

from numpyro import infer


def run_mcmc(
    model_fn,
    model_data,
    verbose: bool = False,
    rng_seed: int = 0xdeadbeef,
):
    sampler = infer.MCMC(
        infer.NUTS(model_fn),
        num_warmup=2000,
        num_samples=10000,
        num_chains=5,
        progress_bar=verbose,
    )

    sampler.run(jax.random.PRNGKey(rng_seed), model_data)

    if verbose:
        sampler.print_summary()

    samples = sampler.get_samples(group_by_chain=False)

    return samples
