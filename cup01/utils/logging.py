import mlflow


def test_mlflow_connection():
    mlflow.set_tracking_uri("http://10.121.252.164:5001")
    mlflow.set_experiment("test_connection")
    with mlflow.start_run():
        mlflow.log_param("test", "test")
        with open("/tmp/test.txt", "w") as f:
            f.write("hello world")
        mlflow.log_artifact("/tmp/test.txt")
        mlflow.get_artifact_uri()
