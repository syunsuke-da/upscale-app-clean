FROM python:3.10-slim

# 依存ツール + OpenCV runtime最小セット
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存インストール
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体
COPY app.py /app/
COPY templates /app/templates

# 本番サーバ起動設定（Cloud Run は $PORT を渡す）
ENV PORT=8080
CMD ["gunicorn","app:app","-w","2","-k","gthread","--threads","4","--timeout","180","-b","0.0.0.0:8080"]
