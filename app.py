									
import os, time, io, uuid
from datetime import datetime, timedelta
from flask import Flask, render_template, request, send_file, abort
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)

@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200

@app.route("/", methods=["GET"])
def index():    return "Hello from Cloud Run", 200

# ルート一覧を返すデバッグ用（動作確認に便利）
@app.route("/__routes", methods=["GET"])
def __routes():
    return "\n".join(sorted([f"{r.rule} -> {sorted(r.methods)}" for r in app.url_map.iter_rules()]))																	

# ---- Optional: Real-ESRGAN (lazy load) ----
_REALSRGAN = None
def load_realesrgan():
    global _REALSRGAN
    if _REALSRGAN is not None:
        return _REALSRGAN
    try:
        from realesrgan import RealESRGAN
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = RealESRGAN(device, scale=4)
        # 初回は自動で重みDL（環境により数分）。キャッシュされます
        model.load_weights('RealESRGAN_x4plus.pth')
        _REALSRGAN = model
        return _REALSRGAN
    except Exception as e:
        print("Real-ESRGANの読み込みに失敗:", e)
        return None

# ローカル実行用（Cloud Run では使われません）
if __name__ == "__main__":
    # 127.0.0.1:8080 で起動（PORT 環境変数があればそれを使う)
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)

# ---- Simple in-memory rate limit ----
WINDOW_SEC = 60
MAX_REQ = 2
usage = {}

def rate_limited(ip):
    now = time.time()
    if ip not in usage:
        usage[ip] = []
    usage[ip] = [t for t in usage[ip] if now - t < WINDOW_SEC]
    if len(usage[ip]) >= MAX_REQ:
        return True
    usage[ip].append(now)
    return False

app = Flask(__name__)

def pil_to_bytes(pil_img, fmt='PNG'):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return buf

def npimg_to_pil(arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode='L')
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def upscale_fast_2x(img_pil):
    arr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    up = cv2.resize(arr, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    return npimg_to_pil(up)

def upscale_realesrgan_4x(img_pil):
    model = load_realesrgan()
    if model is None:
        raise RuntimeError("高品質モデルが未準備です（後で再試行してください）")
    np_img = np.array(img_pil.convert('RGB'))
    sr_img = model.predict(np_img)  # returns numpy RGB
    return Image.fromarray(sr_img)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upscale", methods=["POST"])
def upscale():
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    if rate_limited(client_ip):
        return "利用制限中：1分あたり2回までです。少し待ってから再試行してください。", 429

    if 'image' not in request.files:
        abort(400, "画像ファイルがありません")
    f = request.files['image']
    mode = request.form.get('mode', '2x')

    # サイズ制限（5MB）
    f.seek(0, os.SEEK_END)
    size = f.tell()
    if size > 5 * 1024 * 1024:
        abort(400, "5MB以下の画像のみ対応しています")
    f.seek(0)

    img = Image.open(f.stream).convert("RGB")
    try:
        if mode == '4x':
            out = upscale_realesrgan_4x(img)
            filename = f"upscaled_4x_{uuid.uuid4().hex}.png"
        else:
            out = upscale_fast_2x(img)
            filename = f"upscaled_2x_{uuid.uuid4().hex}.png"
    except Exception as e:
        abort(500, f"処理エラー: {e}")

    buf = pil_to_bytes(out, fmt='PNG')
    return send_file(buf, mimetype="image/png",
                     as_attachment=True, download_name=filename)

if __name__ == "__main__":
    # 本番は gunicorn を推奨: gunicorn app:app -w 2 -k gthread --threads 4 --timeout 120
    app.run(host="0.0.0.0", port=8000, debug=True)
							
	

@app.route("/__version", methods=["GET"])
def __version():
    return os.environ.get("GIT_SHA","unknown"), 200

@app.route("/__routes", methods=["GET"])
def __routes():
    return "\n".join(sorted([f"{r.rule} -> {sorted(r.methods)}" for r in app.url_map.iter_rules()])), 200

# Cloud Run/一般的なLBでよく使うヘルスパスを全部用意
@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200

@app.route("/_ah/health", methods=["GET"])
def ah_health():
    return "ok", 200

@app.route("/health", methods=["GET"])
def health_simple():
    return "ok", 200
