import mlflow
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed


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
    clf, X, y, n_folds=5, seed=42, verbose=1, mlflow_logging=True, n_jobs=-1
):
    if verbose:
        print("Running {}-fold cross-validation".format(n_folds))
    cv_results = {
        "train_auc": np.zeros(n_folds),
        "val_auc": np.zeros(n_folds),
    }
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Helper function to perform the cross-validation for one fold
    def cross_validate_fold(i, train_idx, test_idx):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)

        train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
        val_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

        # Log metrics in mlflow
        if mlflow_logging:
            mlflow.log_metric("cv_train_auc", train_auc, step=i)
            mlflow.log_metric("cv_val_auc", val_auc, step=i)

        if verbose:
            print(
                f"Fold: {i}, train_auc: {train_auc:.3f}, val_auc: {val_auc:.3f}"
            )

        return train_auc, val_auc

    # Running the cross-validation folds in parallel using joblib
    results = Parallel(n_jobs=n_jobs)(
        delayed(cross_validate_fold)(i, train_idx, test_idx)
        for i, (train_idx, test_idx) in enumerate(skf.split(X, y))
    )

    # Collecting results into cv_results
    for i, (train_auc, val_auc) in enumerate(results):
        cv_results["train_auc"][i] = train_auc
        cv_results["val_auc"][i] = val_auc

    return cv_results
