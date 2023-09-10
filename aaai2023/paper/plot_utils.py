
from typing import List, Dict, Tuple, Callable

import numpy as np

from matplotlib import rc_context, pyplot as plt

from aaai2023.paper.util import ExperimentError


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
                    positions=[(group_size + 1)*ix + jx + 1],
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
                    fontsize="x-small",
                )
                plt.annotate(
                    text=f"{mean_data.get_ydata()[0]:.3f}",
                    xy=(mean_data.get_xdata().mean(), mean_data.get_ydata()[0] + .01),
                    xycoords='data',
                    fontsize="x-small",
                )

        ax.set_xticks([
            (group_size + 1) * i + ((group_size + 1) / 2)
            for i in range(len(x_labels))
        ])
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

        plt.close(fig)


def all_box_plots(
    groupings: Dict[Tuple[str, str], List[ExperimentError]],
    x_labels: List[str],
    group_labels: List[str],
    group_style: Dict[str, dict],
    file_name_fn: Callable[[str, str], str],
    breakdown_by_clf: bool = True,
    x_label: str | None = None,
):
    aes = {
        k: np.array([e.absolute_error for e in vs])
        for k, vs in groupings.items()
    }
    apes = {
        k: np.array([e.absolute_percentage_error for e in vs])
        for k, vs in groupings.items()
    }
    nas = {
        k: np.array([e.normalized_absolute_score for e in vs])
        for k, vs in groupings.items()
    }
    sapes = {
        k: np.array([e.symmetric_absolute_percentage_error for e in vs])
        for k, vs in groupings.items()
    }
    box_plots(
        x_labels=x_labels,
        group_labels=group_labels,
        data=aes,
        group_style=group_style,
        x_label=x_label,
        y_label="Absolute Error",
        save_as=file_name_fn("all", "AE"),
    )
    box_plots(
        x_labels=x_labels,
        group_labels=group_labels,
        data=apes,
        group_style=group_style,
        x_label=x_label,
        y_label="Absolute Relative Error",
        save_as=file_name_fn("all", "APE"),
    )
    box_plots(
        x_labels=x_labels,
        group_labels=group_labels,
        data=nas,
        group_style=group_style,
        x_label=x_label,
        y_label="Normalized Absolute Score",
        save_as=file_name_fn("all", "NAS"),
    )
    box_plots(
        x_labels=x_labels,
        group_labels=group_labels,
        data=sapes,
        group_style=group_style,
        x_label=x_label,
        y_label="Symmetric Absolute Percentage Error",
        save_as=file_name_fn("all", "SAPE"),
    )

    if breakdown_by_clf:
        for clf in ['electra', 'cardiffnlp', 'tfidf-svm', 'perspective']:
            aes = {
                k: np.array([
                    e.absolute_error
                    for e in vs
                    if clf in e.scores_file
                ])
                for k, vs in groupings.items()
            }
            apes = {
                k: np.array([
                    e.absolute_percentage_error
                    for e in vs
                    if clf in e.scores_file
                ])
                for k, vs in groupings.items()
            }
            nas = {
                k: np.array([
                    e.normalized_absolute_score
                    for e in vs
                    if clf in e.scores_file
                ])
                for k, vs in groupings.items()
            }
            sapes = {
                k: np.array([
                    e.symmetric_absolute_percentage_error
                    for e in vs
                    if clf in e.scores_file
                ])
                for k, vs in groupings.items()
            }
            box_plots(
                x_labels=x_labels,
                group_labels=group_labels,
                data=aes,
                group_style=group_style,
                x_label=x_label,
                y_label="Absolute Error",
                save_as=file_name_fn(clf, "AE"),
            )
            box_plots(
                x_labels=x_labels,
                group_labels=group_labels,
                data=apes,
                group_style=group_style,
                x_label=x_label,
                y_label="Absolute Relative Error",
                save_as=file_name_fn(clf, "APE"),
            )
            box_plots(
                x_labels=x_labels,
                group_labels=group_labels,
                data=nas,
                group_style=group_style,
                x_label=x_label,
                y_label="Normalized Absolute Score",
                save_as=file_name_fn(clf, "NAS"),
            )
            box_plots(
                x_labels=x_labels,
                group_labels=group_labels,
                data=sapes,
                group_style=group_style,
                x_label=x_label,
                y_label="Symmetric Absolute Percentage Error",
                save_as=file_name_fn("all", "SAPE"),
            )


def bar_plots(
        x_labels: List[str],
        group_labels: List[str],
        data: Dict[str, np.array],
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
        xs_base = np.arange(len(x_labels))

        for group_ix, group in enumerate(group_labels):
            plt.bar(
                (group_size + 1) * xs_base + group_ix,
                data[group],
                label=group,
                **group_style[group],
            )

        ax.set_xticks((group_size + 1) * xs_base + ((group_size - 1) / 2))
        ax.set_xticklabels(x_labels)

        plt.legend(loc="upper right")

        if x_label is not None:
            ax.set_xlabel(xlabel=x_label)

        if y_label is not None:
            ax.set_ylabel(ylabel=y_label)

        if save_as is None:
            plt.show()
        else:
            plt.savefig(save_as)

        plt.close(fig)
