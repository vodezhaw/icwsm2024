
from aaai2023.paper.sampling_fails import main as sampling_fails
from aaai2023.paper.quant_method_boxplots import main as compare_quant_methods
from aaai2023.paper.out_of_domain import main as out_of_domain
from aaai2023.paper.subsampling_plots import main as subsample_prevalence
from aaai2023.paper.n_sample_plots import main as sample_sizes


def main():
    sampling_fails()
    compare_quant_methods()
    out_of_domain()
    subsample_prevalence()
    sample_sizes()


if __name__ == '__main__':
    main()