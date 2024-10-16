import numpy as np
from scipy import sparse


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
