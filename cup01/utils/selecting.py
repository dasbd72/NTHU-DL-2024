from typing import Callable

import mlflow
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def mlflow_cross_validate(
    clf,
    X,
    y,
    columns=None,
    n_folds=5,
    seed=42,
    verbose=1,
    n_jobs=-1,
    mlflow_parent_run: mlflow.ActiveRun = None,
    mlflow_setup: Callable = None,
):
    if mlflow_parent_run is None:
        raise ValueError("mlflow_parent_run must be provided")
    if mlflow_setup is None:
        raise ValueError("mlflow_setup must be provided")

    # Setup mlflow
    mlflow_setup()

    # Start of cross_validate_fold
    # Helper function to perform the cross-validation for one fold
    def cross_validate_fold(i, train_idx, test_idx):
        # Setup mlflow
        mlflow_setup()
        with mlflow.start_run(
            run_name="cross_validate_fold",
            parent_run_id=mlflow_parent_run.info.run_id,
            nested=True,
        ):
            mlflow.log_param("columns", columns)

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            clf.fit(X_train, y_train)

            train_auc = roc_auc_score(
                y_train, clf.predict_proba(X_train)[:, 1]
            )
            val_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

            mlflow.log_metric("train_auc", train_auc)
            mlflow.log_metric("val_auc", val_auc)

            # Log metrics in mlflow
            if verbose:
                print(
                    f"Fold: {i}, train_auc: {train_auc:.5f}, val_auc: {val_auc:.5f}"
                )

        return train_auc, val_auc
        # End of cross_validate_fold

    cv_results = {
        "train_auc": np.zeros(n_folds),
        "val_auc": np.zeros(n_folds),
    }

    mlflow.log_param("columns", columns)
    mlflow.log_param("n_folds", n_folds)
    mlflow.log_param("seed", seed)

    if verbose:
        print("Parent run ID:", mlflow_parent_run.info.run_id)
        print("Running {}-fold cross-validation".format(n_folds))

    # Running the cross-validation folds in parallel using joblib
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = Parallel(n_jobs=n_jobs)(
        delayed(cross_validate_fold)(i, train_idx, test_idx)
        for i, (train_idx, test_idx) in enumerate(skf.split(X, y))
    )

    # Collecting results into cv_results
    for i, (train_auc, val_auc) in enumerate(results):
        cv_results["train_auc"][i] = train_auc
        cv_results["val_auc"][i] = val_auc

    train_auc_mean = cv_results["train_auc"].mean()
    val_auc_mean = cv_results["val_auc"].mean()
    mlflow.log_metric("train_auc", train_auc_mean)
    mlflow.log_metric("val_auc", val_auc_mean)
    if verbose:
        print(
            "mean train_auc: {:.5f}, mean val_auc: {:.5f}".format(
                train_auc_mean, val_auc_mean
            )
        )

    return cv_results
