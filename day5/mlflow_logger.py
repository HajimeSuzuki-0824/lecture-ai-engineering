# mlflow_logger.py
import mlflow

def start_experiment(experiment_name="llama3_rag"):
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()

def log_params(config: dict):
    for section, params in config.items():
        if isinstance(params, dict):
            for k, v in params.items():
                mlflow.log_param(f"{section}.{k}", v)
        else:
            mlflow.log_param(section, params)

def log_metrics(metrics: dict):
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

def log_output(response: str):
    mlflow.log_text(response, "output.txt")
