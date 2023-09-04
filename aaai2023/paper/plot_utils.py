
from typing import List, Dict, Tuple

import numpy as np

from matplotlib import rc_context, pyplot as plt


def box_plots(
    x_labels: List[str],
    group_labels: List[str],
    data: Dict[Tuple[str, str], np.array],
    group_style: Dict[str, dict],
    x_label: str | None = None,
    y_label: str | None = None,
    save_as: str | None = None,
):
    with rc_context({
        "lines.linewidth": 5,
        "font.size": 20,
    }):
        fig, ax = plt.subplots()
        fig.set_size_inches(16, 9)
        fig.set_tight_layout(True)

        group_size = len(group_labels)

        for ix, x_name in enumerate(x_labels):
            for jx, group_name in enumerate(group_labels):
                box_data = plt.boxplot(
                    data[x_name, group_name],
                    positions=[group_size*ix + jx + 1],
                    widths=.6,
                    whis=(5, 95),
                    showmeans=True,
                    meanline=True,
                    boxprops={
                        'linewidth': 5,
                        **group_style[group_name],
                    },
                    whiskerprops={
                        'linewidth': 2,
                    },
                    capprops={
                        'linewidth': 2,
                    },
                    medianprops={
                        'linewidth': 2,
                    },
                    meanprops={
                        'linewidth': 2,
                    },
                )
                median_data = box_data['medians'][0]
                mean_data = box_data['means'][0]
                plt.annotate(
                    text=f"{median_data.get_ydata()[0]:.3f}",
                    xy=(median_data.get_xdata().mean(), median_data.get_ydata()[0] - .05),
                    xycoords='data',
                )
                plt.annotate(
                    text=f"{mean_data.get_ydata()[0]:.3f}",
                    xy=(mean_data.get_xdata().mean(), mean_data.get_ydata()[0] + .05),
                    xycoords='data',
                )

        ax.set_xticks(
            group_size * i + (group_size / 2)
            for i in range(len(x_labels))
        )
        ax.set_xticklabels(x_labels)

        if x_label is not None:
            ax.set_xlabel(xlabel=x_label)

        if y_label is not None:
            ax.set_ylabel(ylabel=y_label)

        if group_size > 1:
            dummy_lines = [
                plt.plot([1, 1], label=group_name, **group_style[group_name])
                for group_name in group_labels
            ]

            plt.legend(loc="upper right")

            for dummy in dummy_lines:
                dummy[0].set_visible(False)

        if save_as is None:
            plt.show()
        else:
            plt.savefig(save_as)
