# ベースイメージ：軽量なPython + CUDA対応（必要に応じて変更可）
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 作業ディレクトリを作成
WORKDIR /app

# 必要なパッケージファイルをコピー
COPY requirements.txt .

# Pythonパッケージをインストール
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    apt-get update && apt-get install -y git

# ソースコード全体をコピー
COPY . .

# Hugging Faceのアクセストークンを環境変数で渡す（build時は空でもOK）
ENV HUGGINGFACE_TOKEN=your_token_here
ENV OPENAI_API_KEY=your_api_key_here
ENV SLACK_WEBHOOK_URL=

# モデルとトークナイザーを事前ダウンロード（キャッシュレイヤー活用）
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
               AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct'); \
               AutoModel.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')"

# 実行コマンド
CMD ["python", "main.py"]
