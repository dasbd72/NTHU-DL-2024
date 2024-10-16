import os
import pickle

from .logging import test_mlflow_connection
from .plotting import plot_auc, plot_correlations
from .selecting import mlflow_cross_validate
from .training import get_batches


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
