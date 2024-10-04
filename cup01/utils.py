import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import mlflow


def dummy(doc):
    return doc


def do_or_load(path, func):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        res = func()
        with open(path, "wb") as f:
            pickle.dump(res, f)
    else:
        with open(path, "rb") as f:
            res = pickle.load(f)
    return res


def get_batches(X, y, batch_size=32, repeated=False):
    length = X.shape[0]
    while True:
        for i in range(0, length, batch_size):
            if isinstance(X, np.ndarray):
                X_batch = X[i : i + batch_size]
            elif isinstance(X, sparse.csr_matrix):
                X_batch = X[i : i + batch_size].toarray()
            else:
                raise ValueError("Invalid X type")
            if y is None:
                yield X_batch
                continue
            y_batch = y[i : i + batch_size]
            yield X_batch, y_batch
        if not repeated:
            break


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


def plot_correlations(X, y):
    correlations = []
    for i in range(X.shape[1]):
        correlations.append(
            (X.columns[i], np.corrcoef(X.values[:, i], y)[0, 1])
        )
    correlations = sorted(correlations, key=lambda x: -abs(x[1]))
    for c in correlations[:10]:
        print(c)
    plt.figure(figsize=(3, 3))
    plt.barh(
        range(X.shape[1]),
        [c[1] for c in reversed(correlations)],
        tick_label=[c[0] for c in reversed(correlations)],
    )
    plt.show()


def test_mlflow_connection():
    mlflow.set_tracking_uri("http://10.121.252.164:5001")
    mlflow.set_experiment("test_connection")
    with mlflow.start_run():
        mlflow.log_param("test", "test")
        with open("/tmp/test.txt", "w") as f:
            f.write("hello world")
        mlflow.log_artifact("/tmp/test.txt")
        mlflow.get_artifact_uri()
    mlflow.delete_experiment("test_connection")
