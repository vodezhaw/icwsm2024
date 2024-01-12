
from pathlib import Path

from icwsm2024.paper.sampling_fails import main as sampling_fails
from icwsm2024.paper.compare_quantification_methods import main as compare_quant_methods
from icwsm2024.paper.out_of_domain import main as out_of_domain
from icwsm2024.paper.low_prevalence import main as subsample_prevalence
from icwsm2024.paper.fewer_samples import main as sample_sizes
from icwsm2024.paper.calibration_methods import main as calibration_methods


def main():
    plot_path = Path('paper_plots/')
    if not plot_path.exists():
        plot_path.mkdir()
    for sub_dir in ['compare_quantification_strategies', 'fails', 'fewer_samples', 'low_prevalence', 'out_of_domain', 'calibration_methods']:
        sub_path = plot_path / sub_dir
        if not sub_path.exists():
            sub_path.mkdir()

    sampling_fails()
    compare_quant_methods()
    out_of_domain()
    subsample_prevalence()
    sample_sizes()
    calibration_methods()


if __name__ == '__main__':
    main()
