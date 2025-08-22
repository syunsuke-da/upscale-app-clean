FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py /app/
COPY templates /app/templates
ENV PORT=8080
CMD ["gunicorn","app:app","-w","2","-k","gthread","--threads","4","--timeout","180","-b","0.0.0.0:8080"]
