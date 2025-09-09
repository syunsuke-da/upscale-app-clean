# app.py — Cloud Run 用の最小＆安定構成

import os
import io
import time
import uuid
import logging
import urllib.request
from datetime import datetime

from flask import Flask, render_template, request, send_file, abort
from PIL import Image
import numpy as np

# ------------ 設定 ------------
logging.basicConfig(level=logging.INFO)

# 5〜6MB くらいに制限（必要に応じて変更）
MAX_UPLOAD_MB = 6
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024

# Real-ESRGAN の重み（Cloud Run では起動後 /tmp にDL）
WEIGHT_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.5.0/RealESRGAN_x4plus.pth"
)
WEIGHT_PATH = "/tmp/RealESRGAN_x4plus.pth"


# ------------ Flask ------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# ------------ ヘルス＆デバッグ ------------
@app.route("/health", methods=["GET"])
@app.route("/_ah/health", methods=["GET"])
def health():
    return "ok", 200


@app.route("/healthz", methods=["GET"])
def healthz():
    # GFE 側404になる環境もあるので、使わなくてもOK
    return "ok", 200


@app.route("/__version", methods=["GET"])
def __version():
    return os.environ.get("GIT_SHA", "unknown"), 200


@app.route("/__routes", methods=["GET"])
def __routes():
    return "\n".join(
        sorted(f"{r.rule} -> {sorted(r.methods)}" for r in app.url_map.iter_rules())
    ), 200


# ------------ レート制限（超簡易） ------------
WINDOW_SEC = 60
MAX_REQ = 2
_usage = {}  # ip -> [timestamps]


def rate_limited(ip: str) -> bool:
    now = time.time()
    _usage.setdefault(ip, [])
    _usage[ip] = [t for t in _usage[ip] if now - t < WINDOW_SEC]
    if len(_usage[ip]) >= MAX_REQ:
        return True
    _usage[ip].append(now)
    return False


# ------------ 画像ユーティリティ ------------
def pil_to_bytes(pil_img: Image.Image, fmt="PNG") -> io.BytesIO:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return buf


def npimg_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    # OpenCVはBGR、PillowはRGB
    import cv2

    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


# ------------ Real-ESRGAN 4x ------------
def ensure_realesrgan_weights() -> str:
    if not os.path.exists(WEIGHT_PATH):
        os.makedirs(os.path.dirname(WEIGHT_PATH), exist_ok=True)
        logging.info("Downloading Real-ESRGAN weights to /tmp ...")
        urllib.request.urlretrieve(WEIGHT_URL, WEIGHT_PATH)
        logging.info("Done downloading weights.")
    return WEIGHT_PATH


_REALSRGAN = None


def load_realesrgan():
    """初回のみロード（CPU 版）"""
    global _REALSRGAN
    if _REALSRGAN is not None:
        return _REALSRGAN
    try:
        from realesrgan import RealESRGAN
        import torch

        device = torch.device("cpu")
        model = RealESRGAN(device, scale=4)
        model.load_weights(ensure_realesrgan_weights())
        _REALSRGAN = model
        return _REALSRGAN
    except Exception:
        logging.exception("load_realesrgan failed")
        return None


# ------------ アップスケール実装 ------------
def upscale_fast_2x(img_pil: Image.Image) -> Image.Image:
    """高速2倍：OpenCV BICUBIC"""
    import cv2

    arr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    up = cv2.resize(arr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    return npimg_to_pil(up)


def upscale_realesrgan_4x(img_pil: Image.Image) -> Image.Image:
    model = load_realesrgan()
    if model is None:
        # ここに来た場合は起動直後で重み未ダウンロード等
        abort(503, "高品質モデルが未準備です（しばらくして再試行）")
    # realesrgan==0.3.0 は PIL.Image を返す
    return model.predict(img_pil)


# ------------ ルーティング ------------
@app.route("/", methods=["GET"])
def index():
    # templates/index.html があればそれを表示、無ければ簡易フォーム
    try:
        return render_template("index.html")
    except Exception:
        return (
            """<!doctype html><html><body>
            <h1>Image Upscaler</h1>
            <form method="post" action="/upscale" enctype="multipart/form-data">
              <p><input type="file" name="image" required></p>
              <p>
                <label><input type="radio" name="mode" value="2x" checked> 2x (fast)</label>
                <label><input type="radio" name="mode" value="4x"> 4x (high quality)</label>
              </p>
              <button type="submit">Upscale</button>
            </form>
            </body></html>""",
            200,
        )


@app.route("/upscale", methods=["POST"])
def upscale():
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    if rate_limited(client_ip):
        return (
            "利用制限中：1分あたり2回までです。少し待ってから再試行してください。",
            429,
        )

    if "image" not in request.files:
        abort(400, "画像ファイルがありません")
    f = request.files["image"]
    if f.filename == "":
        abort(400, "画像ファイル名が空です")

    mode = request.form.get("mode", "2x")

    # サイズ制限チェック
    pos = f.stream.tell()
    f.stream.seek(0, os.SEEK_END)
    size = f.stream.tell()
    f.stream.seek(pos)
    if size > MAX_CONTENT_LENGTH:
        abort(400, f"{MAX_UPLOAD_MB}MB 以下の画像のみ対応しています")

    # 画像読込
    try:
        img = Image.open(f.stream).convert("RGB")
    except Exception:
        abort(400, "画像の読み込みに失敗しました（対応形式：JPG/PNG）")

    t0 = time.time()
    try:
        if mode == "4x":
            out = upscale_realesrgan_4x(img)
            filename = f"upscaled_4x_{uuid.uuid4().hex}.png"
        elif mode == "2x":
            out = upscale_fast_2x(img)
            filename = f"upscaled_2x_{uuid.uuid4().hex}.png"
        else:
            abort(400, "mode は 2x か 4x を指定してください")
    except Exception as e:
        logging.exception("upscale failed: %s", e)
        abort(500, f"処理エラー: {e}")

    buf = pil_to_bytes(out, fmt="PNG")
    logging.info("mode=%s done in %.2fs, size=%d", mode, time.time() - t0, size)
    return send_file(buf, mimetype="image/png", as_attachment=True, download_name=filename)


# ------------ ローカル開発用 ------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
