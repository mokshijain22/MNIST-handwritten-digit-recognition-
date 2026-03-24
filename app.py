from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
CORS(app)

model = load_model('mnist_model.h5')

@app.route('/')
def index():
    return send_file('index.html')


# -------- CANVAS PREDICTION --------
def preprocess_pixels(image_data):
    """
    Expects raw pixel values in range [0, 255] as a flat list/array of 784 values.
    Returns a (1, 28, 28, 1) float32 array ready for model input.
    """
    img = np.array(image_data, dtype='float32').reshape(28, 28)

    # Convert to MNIST style: white digit on black background
    img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    # Small denoise using 3x3 local average
    smoothed = np.copy(img)
    for y in range(28):
        for x in range(28):
            y0, y1 = max(0, y-1), min(28, y+2)
            x0, x1 = max(0, x-1), min(28, x+2)
            smoothed[y, x] = np.mean(img[y0:y1, x0:x1])
    img = np.clip(smoothed, 0.0, 1.0)

    # Crop to bounding box, scale to 20x20, and center in 28x28
    mask = img > 0.05
    if np.any(mask):
        ys, xs = np.where(mask)
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        digit = img[y_min:y_max+1, x_min:x_max+1]

        h, w = digit.shape
        scale = 20.0 / max(h, w)
        new_h = max(1, int(np.round(h * scale)))
        new_w = max(1, int(np.round(w * scale)))

        digit_resized = Image.fromarray((digit * 255).astype('uint8')).resize((new_w, new_h), Image.BILINEAR)
        digit_resized = np.array(digit_resized).astype('float32') / 255.0

        img = np.zeros((28, 28), dtype='float32')
        y_off = (28 - new_h) // 2
        x_off = (28 - new_w) // 2
        img[y_off:y_off+new_h, x_off:x_off+new_w] = digit_resized
    else:
        img = np.zeros((28, 28), dtype='float32')

    return img.reshape(1, 28, 28, 1)


def preprocess_uploaded_image(pil_image):
    """
    Separate preprocessing pipeline for uploaded images.
    Handles images that may be white-on-white (real photos/scans).
    Returns a (1, 28, 28, 1) float32 array ready for model input.
    """
    # Convert to grayscale numpy array in [0, 255]
    img = np.array(pil_image, dtype='float32')

    # Normalize to [0, 1]
    img = img / 255.0

    # Invert if the digit appears to be dark on a light background
    # (MNIST expects white digit on black background)
    if np.mean(img) > 0.5:
        img = 1.0 - img

    # Denoise
    smoothed = np.copy(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            y0, y1 = max(0, y-1), min(img.shape[0], y+2)
            x0, x1 = max(0, x-1), min(img.shape[1], x+2)
            smoothed[y, x] = np.mean(img[y0:y1, x0:x1])
    img = np.clip(smoothed, 0.0, 1.0)

    # Crop to bounding box, scale to 20x20, and center in 28x28
    mask = img > 0.1
    if np.any(mask):
        ys, xs = np.where(mask)
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        digit = img[y_min:y_max+1, x_min:x_max+1]

        h, w = digit.shape
        scale = 20.0 / max(h, w)
        new_h = max(1, int(np.round(h * scale)))
        new_w = max(1, int(np.round(w * scale)))

        digit_resized = Image.fromarray((digit * 255).astype('uint8')).resize(
            (new_w, new_h), Image.LANCZOS
        )
        digit_resized = np.array(digit_resized).astype('float32') / 255.0

        img_out = np.zeros((28, 28), dtype='float32')
        y_off = (28 - new_h) // 2
        x_off = (28 - new_w) // 2
        img_out[y_off:y_off+new_h, x_off:x_off+new_w] = digit_resized
    else:
        img_out = np.zeros((28, 28), dtype='float32')

    return img_out.reshape(1, 28, 28, 1)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']  # Expected: flat list of 784 raw pixel values [0–255]

    image = preprocess_pixels(image_data)

    prediction = model.predict(image, verbose=0)[0]
    top3 = prediction.argsort()[-3:][::-1]

    result = {
        "prediction": int(top3[0]),
        "confidence": float(prediction[top3[0]]),
        "top3": [
            {"digit": int(i), "confidence": float(prediction[i])}
            for i in top3
        ]
    }

    return jsonify(result)


# -------- IMAGE UPLOAD PREDICTION --------
@app.route('/predict_image', methods=['POST'])
def predict_image():
    file = request.files.get('file')

    if file is None:
        return jsonify({"error": "No file uploaded"})

    # Open and convert to grayscale — do NOT resize before preprocessing
    pil_image = Image.open(file).convert('L')

    # Use the dedicated upload pipeline (crop → center → resize to 28x28)
    image = preprocess_uploaded_image(pil_image)

    if np.sum(image) == 0:
        return jsonify({"error": "Empty image after preprocessing"})

    prediction = model.predict(image, verbose=0)[0]
    top3 = prediction.argsort()[-3:][::-1]

    result = {
        "prediction": int(top3[0]),
        "confidence": float(prediction[top3[0]]),
        "top3": [
            {"digit": int(i), "confidence": float(prediction[i])}
            for i in top3
        ]
    }

    if result["confidence"] < 0.5:
        result["warning"] = "Low confidence prediction"

    return jsonify(result)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)