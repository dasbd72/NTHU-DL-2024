import os

import matplotlib.pyplot as plt
import numpy as np
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
    X,
    y,
    file_path=None,
    figsize=None,
    top_n=10,
    print_values=False,
    progress_bar=False,
):
    correlations = []
    if progress_bar:
        for i, column in tqdm(enumerate(X.columns)):
            correlations.append((column, np.corrcoef(X.values[:, i], y)[0, 1]))
    else:
        for i in range(X.shape[1]):
            correlations.append(
                (X.columns[i], np.corrcoef(X.values[:, i], y)[0, 1])
            )
    correlations = sorted(correlations, key=lambda x: -abs(x[1]))[:top_n]
    if print_values:
        for c in correlations:
            print(c)
    plt.figure(figsize=figsize)
    plt.barh(
        range(len(correlations)),
        [c[1] for c in reversed(correlations)],
        tick_label=[c[0] for c in reversed(correlations)],
    )
    if file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)
    plt.show()
