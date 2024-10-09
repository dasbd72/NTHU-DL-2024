import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def plot_auc(train_auc_list, val_auc_list, file_path=None):
    plt.plot(
        range(1, len(train_auc_list) + 1),
        train_auc_list,
        color="blue",
        label="Train auc",
    )
    plt.plot(
        range(1, len(val_auc_list) + 1),
        val_auc_list,
        color="red",
        label="Val auc",
    )
    plt.legend(loc="best")
    plt.xlabel("#Batches")
    plt.ylabel("Auc")
    plt.tight_layout()
    if file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)
    plt.show()


def plot_correlations(
    X: pd.DataFrame,
    y: np.ndarray,
    file_path=None,
    figsize=None,
    top_n=10,
    print_values=False,
):
    correlations = (
        X.corrwith(pd.Series(y))
        .sort_values(
            ascending=False,
            key=lambda x: np.abs(x),
        )
        .head(top_n)
    )

    if print_values:
        print(correlations)

    plt.figure(figsize=figsize)
    plt.barh(correlations.index[::-1], correlations.values[::-1])

    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)

    plt.show()

    return correlations
