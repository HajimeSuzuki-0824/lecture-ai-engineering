from mlflow_logger import start_experiment, log_params, log_metrics, log_output
from notify import notify_slack
import yaml

# config読み込み
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# MLflow実験開始
start_experiment()
log_params(config)

# 実験処理（RAG + Llama推論）
# ...
# response = generate_output(...)
# score = evaluate_answer_accuracy(...)

log_metrics({"answer_score": score})
log_output(response)

# もしスコアが閾値以下ならSlack通知
if score < 2:
    notify_slack(f"⚠️ 回答スコアが低下 ({score})")
