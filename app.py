from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, send_from_directory
import requests
from werkzeug.utils import secure_filename

from predictor import predict_image


ROOT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024
DOWNLOAD_TIMEOUT = 12

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def is_allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(uploaded_file):
    if not is_allowed_file(uploaded_file.filename):
        raise ValueError("Format non supporte. Utilisez JPG, PNG ou WebP.")

    original_name = secure_filename(uploaded_file.filename)
    extension = original_name.rsplit(".", 1)[1].lower()
    filename = f"{uuid4().hex}.{extension}"
    image_path = UPLOAD_DIR / filename
    uploaded_file.save(image_path)
    return image_path, filename


def save_image_from_url(image_url):
    parsed_url = urlparse(image_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("URL invalide. Collez un lien direct vers une image.")

    response = requests.get(
        image_url,
        headers={"User-Agent": "TriIA/1.0"},
        timeout=DOWNLOAD_TIMEOUT,
        stream=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").split(";")[0].lower()
    extension = CONTENT_TYPE_EXTENSIONS.get(content_type)
    if extension is None:
        path_extension = Path(parsed_url.path).suffix.lower().lstrip(".")
        extension = path_extension if path_extension in ALLOWED_EXTENSIONS else None
    if extension is None:
        raise ValueError("Le lien ne pointe pas vers une image JPG, PNG ou WebP.")

    content = bytearray()
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        content.extend(chunk)
        if len(content) > MAX_CONTENT_LENGTH:
            raise ValueError("Image trop lourde. La limite est de 8 Mo.")

    filename = f"{uuid4().hex}.{extension}"
    image_path = UPLOAD_DIR / filename
    image_path.write_bytes(content)
    return image_path, filename


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "Image trop lourde. La limite est de 8 Mo."}), 413


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    uploaded_file = request.files.get("image")
    image_url = request.form.get("image_url", "").strip()
    image_path = None

    try:
        if uploaded_file is not None and uploaded_file.filename != "":
            image_path, filename = save_uploaded_file(uploaded_file)
        elif image_url:
            image_path, filename = save_image_from_url(image_url)
        else:
            return jsonify({"error": "Ajoutez une image ou collez une URL avant l'analyse."}), 400
    except requests.RequestException:
        return jsonify({"error": "Impossible de telecharger cette image. Essayez un autre lien."}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        result = predict_image(image_path)
    except Exception as exc:
        if image_path is not None:
            image_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 500

    result["image_url"] = f"/uploads/{filename}"
    return jsonify(result)


@app.get("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
