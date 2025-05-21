import time
import torch
from model.llama import load_model, generate_output
import yaml

def test_inference_performance():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model, tokenizer = load_model(config["model"]["name"], config["model"]["quant"])

    query = "Inference Time Scalingとは？"
    system_prompt = "日本語で答えてください。"

    start = time.time()
    response = generate_output(model, tokenizer, query, system_prompt)
    end = time.time()

    duration = end - start
    print(f"推論時間: {duration:.2f}秒")
    print(f"出力: {response[:100]}...")

    # 推論時間の閾値（例：30秒以下）
    assert duration < 30, "推論が遅すぎます"

def test_output_not_empty():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model, tokenizer = load_model(config["model"]["name"], config["model"]["quant"])
    query = "推論時に計算リソースを変える手法は？"
    response = generate_output(model, tokenizer, query)
    assert response.strip() != "", "出力が空です"
