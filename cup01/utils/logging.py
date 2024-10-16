import logging

import mlflow


def test_mlflow_connection(mlflow_endpoint: str):
    mlflow_logger = logging.getLogger("mlflow")
    mlflow_logger.setLevel(logging.ERROR)
    mlflow.set_tracking_uri(mlflow_endpoint)
    mlflow.set_experiment("test_connection")
    with mlflow.start_run():
        mlflow.log_param("test", "test")
        with open("/tmp/test.txt", "w") as f:
            f.write("hello world")
        mlflow.log_artifact("/tmp/test.txt")
        mlflow.get_artifact_uri()
