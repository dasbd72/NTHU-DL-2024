import mlflow
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


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


def cross_validate(
    clf, X, y, n_folds=5, seed=42, verbose=1, mlflow_logging=True
):
    if verbose:
        print("Running {}-fold cross-validation".format(n_folds))
    cv_results = {
        "train_auc": np.zeros(n_folds),
        "val_auc": np.zeros(n_folds),
    }
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        cv_results["train_auc"][i] = roc_auc_score(
            y_train, clf.predict_proba(X_train)[:, 1]
        )
        cv_results["val_auc"][i] = roc_auc_score(
            y_test, clf.predict_proba(X_test)[:, 1]
        )
        # Check if running
        if mlflow_logging:
            mlflow.log_metric(
                "cv_train_auc", cv_results["train_auc"][i], step=i
            )
            mlflow.log_metric("cv_val_auc", cv_results["val_auc"][i], step=i)
        if verbose:
            print(
                "Fold: {:d}, train_auc: {:.3f}, val_auc: {:.3f}".format(
                    i, cv_results["train_auc"][i], cv_results["val_auc"][i]
                )
            )
    return cv_results
