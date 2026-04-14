import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from utils import analyze_scene, PLACES365_AVAILABLE


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload size


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    """Return the current model status and capabilities."""
    return jsonify({
        "places365_available": PLACES365_AVAILABLE,
        "models": {
            "imagenet": "MobileNetV2 (pretrained)",
            "places365": "Available" if PLACES365_AVAILABLE else "Not loaded",
        },
        "status": "Ready",
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Please choose an image file."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Upload PNG, JPG, JPEG, BMP, GIF, or WEBP."}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    # Prevent accidental overwrite by appending a number when needed.
    if os.path.exists(save_path):
        name, ext = os.path.splitext(filename)
        counter = 1
        while True:
            candidate = f"{name}_{counter}{ext}"
            candidate_path = os.path.join(app.config["UPLOAD_FOLDER"], candidate)
            if not os.path.exists(candidate_path):
                save_path = candidate_path
                break
            counter += 1

    file.save(save_path)

    try:
        result = analyze_scene(save_path)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Failed to analyze image. Please try another image."}), 500


if __name__ == "__main__":
    app.run(debug=True)
