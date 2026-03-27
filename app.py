from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os
import json
import base64
import google.generativeai as genai
import urllib.request

import cv2
from equation_solver import solve_equation_image

# OpenRouter free vision API
OPENROUTER_API_KEY = "REMOVED"

app = Flask(__name__)
CORS(app)

model = load_model('mnist_model.h5')

# ─────────────────────────────────────────────
# Vision model fallback list
# ─────────────────────────────────────────────

VISION_MODELS = [
    "openrouter/free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "nvidia/llama-3.1-nemotron-nano-8b-v1:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-12b-it:free",
]


def call_openrouter(vision_model, img_b64):
    """Call OpenRouter with a specific model and return parsed JSON response."""
    payload = json.dumps({
        "model": vision_model,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                },
                {
                    "type": "text",
                    "text": (
                        "You are a handwritten math equation recognizer. "
                        "The image shows a handwritten equation on a black background with white strokes.\n\n"
                        "1. Read every symbol carefully: digits 0-9, operators +, -, x, /, ^, sqrt, (, ).\n"
                        "2. Reconstruct the full equation.\n"
                        "3. Evaluate it and compute the result.\n\n"
                        "Respond ONLY with valid JSON, no markdown, no backticks, nothing else:\n"
                        "{\"equation\": \"3 + 4\", \"result\": \"7\", \"tokens\": [{\"symbol\": \"3\", \"type\": \"digit\"}, {\"symbol\": \"+\", \"type\": \"operator\"}, {\"symbol\": \"4\", \"type\": \"digit\"}]}"
                    )
                }
            ]
        }]
    }).encode('utf-8')

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Math Recognizer"
        },
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))


# ─────────────────────────────────────────────
# Shared preprocessing helpers
# ─────────────────────────────────────────────

def preprocess_pixels(image_data):
    """
    Expects raw pixel values [0–255] as a flat list of 784 values.
    Returns (1, 28, 28, 1) float32 array ready for model.predict().
    """
    img = np.array(image_data, dtype='float32').reshape(28, 28)
    img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    # 3×3 local average denoise
    smoothed = np.copy(img)
    for y in range(28):
        for x in range(28):
            y0, y1 = max(0, y - 1), min(28, y + 2)
            x0, x1 = max(0, x - 1), min(28, x + 2)
            smoothed[y, x] = np.mean(img[y0:y1, x0:x1])
    img = np.clip(smoothed, 0.0, 1.0)

    # Crop → scale to 20×20 → center in 28×28
    mask = img > 0.05
    if np.any(mask):
        ys, xs = np.where(mask)
        digit = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w  = digit.shape
        scale = 20.0 / max(h, w)
        new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        digit_resized = np.array(
            Image.fromarray((digit * 255).astype('uint8')).resize((new_w, new_h), Image.BILINEAR)
        ).astype('float32') / 255.0
        img = np.zeros((28, 28), dtype='float32')
        y_off, x_off = (28 - new_h) // 2, (28 - new_w) // 2
        img[y_off:y_off + new_h, x_off:x_off + new_w] = digit_resized
    else:
        img = np.zeros((28, 28), dtype='float32')

    return img.reshape(1, 28, 28, 1)


def preprocess_uploaded_image(pil_image):
    """
    Preprocessing for uploaded single-digit images.
    Handles white-on-light-background (real photos/scans).
    Returns (1, 28, 28, 1) float32 array.
    """
    img = np.array(pil_image, dtype='float32') / 255.0
    if np.mean(img) > 0.5:
        img = 1.0 - img  # invert to white-on-black

    smoothed = np.copy(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            y0, y1 = max(0, y - 1), min(img.shape[0], y + 2)
            x0, x1 = max(0, x - 1), min(img.shape[1], x + 2)
            smoothed[y, x] = np.mean(img[y0:y1, x0:x1])
    img = np.clip(smoothed, 0.0, 1.0)

    mask = img > 0.1
    if np.any(mask):
        ys, xs = np.where(mask)
        digit = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w  = digit.shape
        scale = 20.0 / max(h, w)
        new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        digit_resized = np.array(
            Image.fromarray((digit * 255).astype('uint8')).resize((new_w, new_h), Image.LANCZOS)
        ).astype('float32') / 255.0
        img_out = np.zeros((28, 28), dtype='float32')
        y_off, x_off = (28 - new_h) // 2, (28 - new_w) // 2
        img_out[y_off:y_off + new_h, x_off:x_off + new_w] = digit_resized
    else:
        img_out = np.zeros((28, 28), dtype='float32')

    return img_out.reshape(1, 28, 28, 1)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return send_file('index.html')


# ── Single digit from canvas (pixel array) ──
@app.route('/predict', methods=['POST'])
def predict():
    data       = request.get_json()
    image      = preprocess_pixels(data['image'])
    prediction = model.predict(image, verbose=0)[0]
    top3       = prediction.argsort()[-3:][::-1]
    return jsonify({
        "prediction": int(top3[0]),
        "confidence": float(prediction[top3[0]]),
        "top3": [{"digit": int(i), "confidence": float(prediction[i])} for i in top3],
    })


# ── Single digit from uploaded file ──
@app.route('/predict_image', methods=['POST'])
def predict_image():
    file = request.files.get('file')
    if file is None:
        return jsonify({"error": "No file uploaded"})

    pil_image  = Image.open(file).convert('L')
    image      = preprocess_uploaded_image(pil_image)

    if np.sum(image) == 0:
        return jsonify({"error": "Empty image after preprocessing"})

    prediction = model.predict(image, verbose=0)[0]
    top3       = prediction.argsort()[-3:][::-1]
    result     = {
        "prediction": int(top3[0]),
        "confidence": float(prediction[top3[0]]),
        "top3": [{"digit": int(i), "confidence": float(prediction[i])} for i in top3],
    }
    if result["confidence"] < 0.5:
        result["warning"] = "Low confidence prediction"
    return jsonify(result)


# ── Multi-digit prediction from drawn/uploaded image ─────
@app.route('/predict_multidigit', methods=['POST'])
def predict_multidigit():
    file = request.files.get('file')
    if file is None:
        return jsonify({
            "error": "No file uploaded",
            "number": "",
            "digit_count": 0,
            "avg_confidence": 0.0,
            "digits": [],
            "has_low_conf": False
        })

    pil_image = Image.open(file).convert('L')
    img = np.array(pil_image, dtype=np.uint8)

    # Normalize to black background, white strokes
    if np.mean(img) > 127:
        img = 255 - img

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    result = segment_and_predict_digits(binary)
    return jsonify(result)


# ── Equation from canvas (OpenRouter free Vision) ─────
@app.route('/predict_equation_openrouter', methods=['POST'])
def predict_equation_openrouter(raw_data=None):
    try:
        if raw_data is not None:
            raw = raw_data
        elif 'file' in request.files:
            raw = request.files['file'].read()
        else:
            raw = request.data

        if not raw:
            return jsonify({"error": "No image data received", "expression": "", "result": None})

        img_b64 = base64.b64encode(raw).decode('utf-8')

        # Try each model in order until one works
        data = None
        last_error = None
        for vision_model in VISION_MODELS:
            try:
                data = call_openrouter(vision_model, img_b64)
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode('utf-8')
                last_error = f"Model {vision_model} failed {e.code}: {body}"
                if e.code == 404:
                    continue   # model doesn't exist, try next
                elif e.code == 429:
                    continue   # rate limited, try next model
                else:
                    break      # unexpected error, stop
            except Exception as e:
                last_error = str(e)
                continue

        if data is None:
            return jsonify({"error": last_error, "expression": "", "result": None})

        text = data['choices'][0]['message']['content'].strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            if text.startswith("json"):
                text = text[4:]

        parsed = json.loads(text.strip())

        tokens = []
        for t in parsed.get("tokens", []):
            tokens.append({
                "symbol":     t.get("symbol", "?"),
                "type":       t.get("type", "digit"),
                "confidence": 1.0,
                "bbox":       [0, 0, 0, 0]
            })

        raw_result = parsed.get("result", "?")
        try:
            result_val = float(raw_result)
            if result_val == int(result_val):
                result_val = int(result_val)
        except (ValueError, TypeError):
            result_val = raw_result

        return jsonify({
            "expression": parsed.get("equation", ""),
            "result":     result_val,
            "tokens":     tokens,
            "boxes":      [],
            "error":      None
        })

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Model returned invalid JSON: {str(e)}", "expression": "", "result": None})
    except Exception as e:
        return jsonify({"error": str(e), "expression": "", "result": None})


# ── Multi-digit segmentation helper ──────────

def segment_and_predict_digits(binary: np.ndarray) -> dict:
    """
    Given a white-on-black binary image (uint8, 0/255), detect all digit
    contours, sort them left-to-right, predict each one with the CNN, and
    return a structured result.
    """
    h_img, w_img = binary.shape

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"error": "Empty canvas — nothing to recognise", "number": "", "digits": []}

    min_area = max(30, int(w_img * h_img * 0.003))
    raw_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area >= min_area:
            raw_boxes.append([x, y, w, h])

    if not raw_boxes:
        return {"error": "Only noise detected — draw more clearly", "number": "", "digits": []}

    pad = max(3, w_img // 60)

    def expand(box):
        x, y, w, h = box
        return [max(0, x - pad), max(0, y - pad),
                min(w_img, x + w + pad), min(h_img, y + h + pad)]

    def overlaps(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

    boxes_xyxy = [expand(b) for b in raw_boxes]

    changed = True
    while changed:
        changed = False
        merged = []
        used = [False] * len(boxes_xyxy)
        for i in range(len(boxes_xyxy)):
            if used[i]:
                continue
            cur = list(boxes_xyxy[i])
            for j in range(i + 1, len(boxes_xyxy)):
                if used[j]:
                    continue
                if overlaps(cur, boxes_xyxy[j]):
                    cur[0] = min(cur[0], boxes_xyxy[j][0])
                    cur[1] = min(cur[1], boxes_xyxy[j][1])
                    cur[2] = max(cur[2], boxes_xyxy[j][2])
                    cur[3] = max(cur[3], boxes_xyxy[j][3])
                    used[j] = True
                    changed = True
            merged.append(cur)
            used[i] = True
        boxes_xyxy = merged

    boxes_xywh = []
    for x1, y1, x2, y2 in boxes_xyxy:
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_img, x2); y2 = min(h_img, y2)
        w = x2 - x1; h = y2 - y1
        if w > 0 and h > 0:
            boxes_xywh.append((x1, y1, w, h))

    boxes_xywh.sort(key=lambda b: b[0])

    digits = []
    for idx, (x, y, w, h) in enumerate(boxes_xywh):
        roi = binary[y: y + h, x: x + w]

        if roi.size == 0 or np.sum(roi) == 0:
            continue

        border = max(4, int(max(w, h) * 0.1))
        roi_bordered = cv2.copyMakeBorder(
            roi, border, border, border, border, cv2.BORDER_CONSTANT, value=0
        )

        rh, rw = roi_bordered.shape
        scale  = 20.0 / max(rh, rw)
        new_h  = max(1, int(round(rh * scale)))
        new_w  = max(1, int(round(rw * scale)))

        roi_scaled = cv2.resize(
            roi_bordered, (new_w, new_h), interpolation=cv2.INTER_AREA
        )

        canvas = np.zeros((28, 28), dtype=np.float32)
        y_off  = (28 - new_h) // 2
        x_off  = (28 - new_w) // 2
        canvas[y_off: y_off + new_h, x_off: x_off + new_w] = roi_scaled.astype(np.float32) / 255.0

        inp  = canvas.reshape(1, 28, 28, 1)
        prob = model.predict(inp, verbose=0)[0]
        pred = int(np.argmax(prob))
        conf = float(prob[pred])

        digits.append({
            "index":      idx,
            "digit":      pred,
            "confidence": round(conf, 4),
            "bbox":       [x, y, w, h],
            "low_conf":   conf < 0.50,
        })

    if not digits:
        return {"error": "Could not recognise any digits", "number": "", "digits": []}

    number_str = "".join(str(d["digit"]) for d in digits)
    avg_conf   = round(sum(d["confidence"] for d in digits) / len(digits), 4)

    return {
        "number":           number_str,
        "digit_count":      len(digits),
        "avg_confidence":   avg_conf,
        "digits":           digits,
        "has_low_conf":     any(d["low_conf"] for d in digits),
        "error":            None,
    }


# ── Gemini Vision — free, 1500 requests/day ───────────────
GEMINI_API_KEY = "REMOVED"  # paste your key
genai.configure(api_key=GEMINI_API_KEY)

@app.route('/predict_equation', methods=['POST'])
def predict_equation():
    try:
        if 'file' in request.files:
            raw = request.files['file'].read()
        else:
            raw = request.data

        if not raw:
            return jsonify({"error": "No image data received", "expression": "", "result": None})

        if not raw:
            return jsonify({"error": "No image data received", "expression": "", "result": None})

        image_part = {"mime_type": "image/png", "data": raw}
        prompt = (
            "You are a handwritten math equation recognizer. "
            "The image shows a handwritten equation on a black background with white strokes.\n\n"
            "1. Read every symbol carefully: digits 0-9, operators +, -, x, /, ^, (, ).\n"
            "2. Reconstruct the full equation.\n"
            "3. Evaluate it and compute the result.\n\n"
            "Respond ONLY with valid JSON, no markdown, no backticks:\n"
            '{"equation": "3 + 4", "result": "7", "tokens": '
            '[{"symbol": "3", "type": "digit"}, '
            '{"symbol": "+", "type": "operator"}, '
            '{"symbol": "4", "type": "digit"}]}'
        )

        last_error = None

        # Try explicitly supported models first.
        gemini_candidates = ["gemini-1.5-mini", "gemini-1.5-pro", "gemini-1.0-preview"]

        # If possible, query the API for available models to avoid hardcoding unsupported names.
        try:
            available = genai.list_models() if hasattr(genai, 'list_models') else []
            if isinstance(available, dict) and 'models' in available:
                available = [m.get('name', '') for m in available['models']]
            gemini_candidates = [m for m in gemini_candidates if m in available] or gemini_candidates
        except Exception:
            # ignore list_models failures; continue with defaults
            pass

        for model_name in gemini_candidates:
            try:
                gemini_model = genai.GenerativeModel(model_name)
                response = gemini_model.generate_content([prompt, image_part])
                text = response.text.strip()

                if text.startswith("```"):
                    parts = text.split("```")
                    text = parts[1] if len(parts) > 1 else text
                    if text.startswith("json"):
                        text = text[4:]

                parsed = json.loads(text.strip())
                break
            except Exception as e:
                last_error = str(e)
                if "not found" in last_error.lower() or "404" in last_error:
                    continue
                return jsonify({"error": last_error, "expression": "", "result": None})

        else:
            # Gemini model not available; fallback to OpenRouter pipeline.
            return predict_equation_openrouter(raw_data=raw)

        tokens = [
            {"symbol": t.get("symbol","?"), "type": t.get("type","digit"),
             "confidence": 1.0, "bbox": [0,0,0,0]}
            for t in parsed.get("tokens", [])
        ]

        raw_result = parsed.get("result", "?")
        try:
            result_val = float(raw_result)
            if result_val == int(result_val):
                result_val = int(result_val)
        except (ValueError, TypeError):
            result_val = raw_result

        return jsonify({
            "expression": parsed.get("equation", ""),
            "result":     result_val,
            "tokens":     tokens,
            "boxes":      [],
            "error":      None
        })

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON from Gemini: {str(e)}", "expression": "", "result": None})
    except Exception as e:
        return jsonify({"error": str(e), "expression": "", "result": None})    # ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)