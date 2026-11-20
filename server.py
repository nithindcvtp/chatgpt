import base64
import io
import json
import os
import time
import zipfile

import requests
from flask import Flask, Response, request, send_from_directory

BASE = "https://integrate.api.nvidia.com/v1"
API_KEY = os.environ.get("NVIDIA_API_KEY") or "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"  # paste your key between the quotes
DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
SKIP = ("embed", "rerank", "retriev", "guard", "safety", "parse", "clip", "riva", "voice")
# If the chosen model fails or stays silent, try models whose IDs contain these, in order.
FALLBACKS = ("kimi-k2-instruct", "deepseek-v3", "glm-4", "step-3.5", "mistral-large-3", "qwen3")
FIRST_TOKEN_TIMEOUT = 45  # seconds of silence before moving to the next model
MAX_ATTEMPTS = 4

MAX_FILE_MB = 10
MAX_TEXT_CHARS = 120_000
MAX_TOTAL_CHARS = 300_000
ZIP_MAX_FILES = 60
ZIP_MAX_TOTAL_MB = 50
IMAGE_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
               "gif": "image/gif", "webp": "image/webp"}

app = Flask(__name__, static_folder=".")
_cache = {"ids": None, "at": 0}


def line(kind, value):
    return json.dumps({"t": kind, "v": value}) + "\n"


def model_ids():
    if _cache["ids"] is not None and time.time() - _cache["at"] < 600:
        return _cache["ids"]
    r = requests.get(f"{BASE}/models", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=15)
    r.raise_for_status()
    ids = sorted(m["id"] for m in r.json().get("data", [])
                 if not any(w in m["id"].lower() for w in SKIP))
    _cache.update(ids=ids, at=time.time())
    return ids


@app.get("/")
def index():
    return send_from_directory(".", "index.html")


@app.get("/api/models")
def models():
    try:
        return {"models": model_ids(), "default": DEFAULT_MODEL}
    except Exception as e:
        return {"models": [], "default": DEFAULT_MODEL, "error": str(e)}, 502


def candidates(first):
    order = [first]
    try:
        ids = model_ids()
    except Exception:
        return order
    for key in FALLBACKS:
        for i in ids:
            if key in i and i not in order:
                order.append(i)
                break
    return order[:MAX_ATTEMPTS]


def read_one(name, data):
    """Return ('text', str) / ('image', part) or raise ValueError with a readable reason."""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext in IMAGE_TYPES:
        b64 = base64.b64encode(data).decode()
        return "image", {"type": "image_url", "image_url": {"url": f"data:{IMAGE_TYPES[ext]};base64,{b64}"}}
    if ext == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("PDF support needs: pip install pypdf")
        try:
            body = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
        except Exception as e:
            raise ValueError(f"could not read PDF ({e})")
        if not body.strip():
            raise ValueError("PDF has no selectable text")
        return "text", body
    try:
        return "text", data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("not a text, PDF or image file")


def expand_zip(name, data):
    """Yield (path, bytes) for files inside a zip, in memory only. Returns (items, skipped)."""
    items, skipped, total = [], [], 0
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError(f"{name} is not a valid zip file.")
    for info in z.infolist():
        p = info.filename
        if info.is_dir() or p.startswith("__MACOSX/") or "/." in "/" + p:
            continue
        if p.lower().endswith(".zip"):
            skipped.append(f"{p} (nested zip)")
            continue
        if len(items) >= ZIP_MAX_FILES:
            skipped.append(f"{p} (over the {ZIP_MAX_FILES}-file limit)")
            continue
        total += info.file_size
        if info.file_size > MAX_FILE_MB * 1024 * 1024 or total > ZIP_MAX_TOTAL_MB * 1024 * 1024:
            skipped.append(f"{p} (too large)")
            continue
        items.append((f"{name}/{p}", z.read(info)))
    return items, skipped


def build_content(query, files):
    """Turn the question plus attached files into one message. Returns (content, error)."""
    files = [f for f in files if f.filename]
    if not files:
        return query, None
    items, skipped, from_zip = [], [], set()
    for f in files:
        name = f.filename or "file"
        data = f.read()
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            return None, f"{name} is larger than {MAX_FILE_MB} MB."
        if name.lower().endswith(".zip"):
            try:
                inner, sk = expand_zip(name, data)
            except ValueError as e:
                return None, str(e)
            items += inner
            skipped += sk
            from_zip.update(n for n, _ in inner)
        else:
            items.append((name, data))

    text_parts, images, budget = [], [], MAX_TOTAL_CHARS
    for name, data in items:
        try:
            kind, value = read_one(name, data)
        except ValueError as e:
            if name in from_zip:
                skipped.append(f"{name} ({e})")
                continue
            return None, f"{name}: {e}."
        if kind == "image":
            images.append(value)
            continue
        if budget <= 0:
            skipped.append(f"{name} (text limit reached)")
            continue
        cut = value[:min(MAX_TEXT_CHARS, budget)]
        budget -= len(cut)
        text_parts.append((name, cut + ("\n[truncated]" if len(cut) < len(value) else "")))

    if not text_parts and not images:
        return None, "No readable files found." + (f" Skipped: {', '.join(skipped[:10])}" if skipped else "")

    prompt = query
    if text_parts:
        blocks = "\n\n".join(f"=== File: {n} ===\n{t}" for n, t in text_parts)
        note = f"\n\n(Files that could not be read: {', '.join(skipped[:20])})" if skipped else ""
        prompt = f"{blocks}{note}\n\n=== Question ===\n{query}"
    if images:
        return [{"type": "text", "text": prompt}] + images, None
    return prompt, None


@app.post("/api/ask")
def ask():
    if request.files or request.form:
        query = request.form.get("query", "").strip()
        first = request.form.get("model") or DEFAULT_MODEL
        files = request.files.getlist("files")
    else:
        body = request.get_json(silent=True) or {}
        query, first, files = body.get("query", "").strip(), body.get("model") or DEFAULT_MODEL, []
    if not query:
        return Response(line("error", "Type a question first."), mimetype="application/x-ndjson")
    content, err = build_content(query, files)
    if err:
        return Response(line("error", err), mimetype="application/x-ndjson")

    if not API_KEY:
        return Response(line("error", "Set the NVIDIA_API_KEY environment variable and restart the server."),
                        mimetype="application/x-ndjson")
    try:
        raw = request.form.get("history") or "[]"
        history = [{"role": m["role"], "content": str(m["content"])} for m in json.loads(raw)[-20:]
                   if m.get("role") in ("user", "assistant") and m.get("content")]
    except Exception:
        history = []
    headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "text/event-stream"}

    def generate():
        last_error = "No model answered."
        for n, model in enumerate(candidates(first)):
            if n:
                yield line("status", f"Trying {model}…")
            else:
                yield line("status", f"Request sent to {model}…")
            payload = {"model": model, "stream": True, "max_tokens": 8192,
                       "messages": history + [{"role": "user", "content": content}]}
            started = False
            try:
                with requests.post(f"{BASE}/chat/completions", headers=headers, json=payload,
                                   stream=True, timeout=(10, FIRST_TOKEN_TIMEOUT)) as r:
                    print(f"{model}: HTTP {r.status_code}")
                    if r.status_code != 200:
                        last_error = f"{model}: HTTP {r.status_code} {r.text[:200]}"
                        continue
                    yield line("status", f"{model} connected. Waiting for the first token…")
                    for raw in r.iter_lines():
                        if not raw or not raw.startswith(b"data: "):
                            continue
                        data = raw[6:]
                        if data == b"[DONE]":
                            break
                        try:
                            delta = json.loads(data)["choices"][0]["delta"]
                        except (KeyError, IndexError, ValueError):
                            continue
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        if reasoning:
                            started = True
                            yield line("reasoning", reasoning)
                        if delta.get("content"):
                            started = True
                            yield line("content", delta["content"])
                if started:
                    return
                last_error = f"{model}: finished without an answer."
            except requests.RequestException as e:
                print(f"{model}: {e}")
                if started:
                    yield line("error", f"Connection lost mid-answer: {e}")
                    return
                last_error = f"{model}: no response within {FIRST_TOKEN_TIMEOUT}s ({type(e).__name__})."
        yield line("error", "No model answered. Last problem: " + last_error)

    resp = Response(generate(), mimetype="application/x-ndjson")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


if __name__ == "__main__":
    app.run(port=5000, debug=False, threaded=True)
