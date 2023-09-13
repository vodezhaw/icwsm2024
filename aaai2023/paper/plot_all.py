
from aaai2023.paper.sampling_fails import main as sampling_fails
from aaai2023.paper.compare_quantification_methods import main as compare_quant_methods
from aaai2023.paper.out_of_domain import main as out_of_domain
from aaai2023.paper.low_prevalence import main as subsample_prevalence
from aaai2023.paper.fewer_samples import main as sample_sizes


def main():
    sampling_fails()
    compare_quant_methods()
    out_of_domain()
    subsample_prevalence()
    sample_sizes()


if __name__ == '__main__':
    main()
