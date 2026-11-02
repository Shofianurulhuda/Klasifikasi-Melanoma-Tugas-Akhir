import io
import base64
import warnings
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload


# =============================================================================
# CONFIG
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sesuaikan path model Anda
MODEL_PATH = Path(r"D:\TUGAS AKHIR\model\Model Output\mobilev2_original.pth")

# Threshold ini bukan untuk mendeteksi gambar salah/non-dermoskopi.
# Threshold hanya digunakan untuk memberi peringatan jika confidence model rendah.
CONFIDENCE_THRESHOLD = 0.70

CLASS_NAMES = {
    0: "Benign",
    1: "Malignant"
}

CLASS_LABELS_ID = {
    0: "Jinak",
    1: "Ganas"
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =============================================================================
# MODEL CNN
# =============================================================================
def build_mobilenetv2():
    model = models.mobilenet_v2(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p=0.2),
        nn.Linear(256, 2),
    )

    return model


_model_cache = None


def get_model():
    global _model_cache

    if _model_cache is not None:
        return _model_cache, None

    if not MODEL_PATH.exists():
        return None, f"File model tidak ditemukan: {MODEL_PATH}"

    try:
        model = build_mobilenetv2()

        checkpoint = torch.load(str(MODEL_PATH), map_location=DEVICE)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()

        _model_cache = model

        return model, None

    except Exception as e:
        return None, str(e)


# =============================================================================
# ROUTES
# =============================================================================
@app.route("/")
def index():
    device_str = str(DEVICE).upper()
    return render_template("index.html", device=device_str)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({
            "error": "Tidak ada file yang diunggah."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "Nama file kosong."
        }), 400

    allowed = {"jpg", "jpeg", "png"}
    ext = file.filename.rsplit(".", 1)[-1].lower()

    if ext not in allowed:
        return jsonify({
            "error": f"Format '{ext}' tidak didukung. Gunakan JPG, JPEG, atau PNG."
        }), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Encode preview untuk ditampilkan di frontend
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Load model
        model, err = get_model()

        if err:
            return jsonify({
                "error": err
            }), 500

        # Inference
        tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        is_low_confidence = confidence < CONFIDENCE_THRESHOLD

        return jsonify({
            "class_idx": pred_idx,
            "class_name": CLASS_NAMES[pred_idx],
            "class_label_id": CLASS_LABELS_ID[pred_idx],
            "confidence": confidence,
            "is_low_confidence": is_low_confidence,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "warning": (
                "Confidence prediksi berada di bawah 70%. "
                "Hasil perlu diinterpretasikan dengan hati-hati."
                if is_low_confidence else ""
            ),
            "probabilities": {
                "benign": float(probs[0]),
                "malignant": float(probs[1])
            },
            "image_preview": img_b64
        })

    except Exception as e:
        return jsonify({
            "error": f"Error saat inferensi: {str(e)}"
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)