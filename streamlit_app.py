from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import requests
import streamlit as st

from predictor import predict_image


ROOT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}
MAX_FILE_SIZE = 8 * 1024 * 1024

CATEGORY_COLORS = {
    "cardboard": "#a86435",
    "glass": "#2f7ebd",
    "metal": "#69727a",
    "paper": "#d8a225",
    "plastic": "#247a4b",
    "trash": "#d7634f",
}


st.set_page_config(
    page_title="Tri IA - Classification des dechets",
    page_icon="♻",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles():
    st.markdown(
        """
        <style>
          :root {
            --bg: #f5f7f2;
            --surface: #ffffff;
            --ink: #17211b;
            --muted: #66716b;
            --line: #dfe7df;
            --green: #247a4b;
            --green-dark: #145936;
            --yellow: #f0b33d;
            --blue: #3e7fb8;
            --red: #d7634f;
          }

          .stApp {
            background:
              linear-gradient(135deg, rgba(36, 122, 75, 0.12), transparent 34%),
              linear-gradient(315deg, rgba(62, 127, 184, 0.12), transparent 38%),
              var(--bg);
            color: var(--ink);
          }

          .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
          }

          h1, h2, h3, p {
            letter-spacing: 0 !important;
          }

          .hero {
            display: grid;
            grid-template-columns: 1.05fr 0.95fr;
            gap: 1.6rem;
            align-items: stretch;
            padding: 2rem;
            border: 1px solid rgba(36, 122, 75, 0.18);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 18px 50px rgba(23, 33, 27, 0.12);
            backdrop-filter: blur(16px);
            margin-bottom: 1.5rem;
          }

          .eyebrow {
            color: var(--green);
            font-size: 0.78rem;
            font-weight: 900;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
          }

          .hero-title {
            font-size: clamp(3.2rem, 8vw, 6.8rem);
            line-height: 0.92;
            font-weight: 950;
            margin: 0;
          }

          .lead {
            color: #39453e;
            font-size: 1.1rem;
            line-height: 1.65;
            max-width: 620px;
            margin-top: 1.3rem;
          }

          .chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 1.4rem;
          }

          .chip {
            padding: 0.6rem 0.85rem;
            border: 1px solid var(--line);
            border-radius: 999px;
            background: #fff;
            color: #334039;
            font-weight: 800;
          }

          .bins {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            align-content: center;
          }

          .bin {
            min-height: 104px;
            display: grid;
            place-items: center;
            border-radius: 8px;
            color: white;
            font-weight: 950;
            box-shadow: inset 0 -18px 0 rgba(0, 0, 0, 0.14);
          }

          .panel {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--surface);
            box-shadow: 0 12px 32px rgba(23, 33, 27, 0.08);
            padding: 1.25rem;
            min-height: 100%;
          }

          .panel-title {
            font-size: 1.1rem;
            font-weight: 950;
            margin: 0 0 0.35rem;
          }

          .panel-copy {
            color: var(--muted);
            margin: 0 0 1rem;
            line-height: 1.55;
          }

          div[data-testid="stFileUploader"] {
            border: 2px dashed #b8c7bb;
            border-radius: 8px;
            background: #f8fbf8;
            padding: 0.8rem;
          }

          div[data-testid="stFileUploader"]:hover {
            border-color: var(--green);
            background: #eef8f1;
          }

          .stButton > button {
            min-height: 48px;
            width: 100%;
            border: 0;
            border-radius: 8px;
            background: var(--green);
            color: white;
            font-weight: 950;
            box-shadow: 0 10px 22px rgba(36, 122, 75, 0.18);
            transition: background 160ms ease, transform 160ms ease;
          }

          .stButton > button:hover {
            background: var(--green-dark);
            color: white;
            transform: translateY(-1px);
          }

          .result-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #fbfdfb;
            padding: 1.2rem;
          }

          .result-label {
            font-size: clamp(2.1rem, 5vw, 4.2rem);
            line-height: 1;
            font-weight: 950;
            margin: 0;
          }

          .confidence-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-top: 1.2rem;
            font-weight: 900;
          }

          .meter {
            height: 13px;
            overflow: hidden;
            border-radius: 999px;
            background: #e7ede7;
            margin-top: 0.55rem;
          }

          .meter-fill {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, var(--yellow), var(--green));
          }

          .tip {
            color: #39453e;
            line-height: 1.6;
            margin-top: 1rem;
          }

          .top-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 1rem;
            align-items: center;
            padding: 0.8rem;
            margin-top: 0.65rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: white;
            font-weight: 900;
          }

          .top-row strong {
            color: var(--green);
          }

          .empty-state {
            min-height: 355px;
            display: grid;
            place-items: center;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #eef3ef;
            color: #7b877f;
            font-weight: 950;
          }

          @media (max-width: 900px) {
            .hero {
              grid-template-columns: 1fr;
              padding: 1.2rem;
            }

            .bins {
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def save_uploaded_file(uploaded_file):
    extension = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Format non supporte. Utilisez JPG, PNG ou WebP.")

    content = uploaded_file.getvalue()
    if len(content) > MAX_FILE_SIZE:
        raise ValueError("Image trop lourde. La limite est de 8 Mo.")

    filename = f"{uuid4().hex}.{extension}"
    image_path = UPLOAD_DIR / filename
    image_path.write_bytes(content)
    return image_path


def save_image_from_url(image_url):
    parsed_url = urlparse(image_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("URL invalide. Collez un lien direct vers une image.")

    response = requests.get(
        image_url,
        headers={"User-Agent": "TriIA/1.0"},
        timeout=12,
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
        if len(content) > MAX_FILE_SIZE:
            raise ValueError("Image trop lourde. La limite est de 8 Mo.")

    filename = f"{uuid4().hex}.{extension}"
    image_path = UPLOAD_DIR / filename
    image_path.write_bytes(content)
    return image_path


def render_hero():
    st.markdown(
        """
        <section class="hero">
          <div>
            <div class="eyebrow">Classification d'images par IA</div>
            <h1 class="hero-title">Tri IA</h1>
            <p class="lead">
              Analysez une vraie photo de dechet depuis votre ordinateur ou depuis une URL,
              puis obtenez une prediction claire avec le niveau de confiance du modele.
            </p>
            <div class="chips">
              <span class="chip">6 categories</span>
              <span class="chip">Upload ou URL</span>
              <span class="chip">TensorFlow local</span>
            </div>
          </div>
          <div class="bins">
            <div class="bin" style="background:#a86435;">Carton</div>
            <div class="bin" style="background:#2f7ebd;">Verre</div>
            <div class="bin" style="background:#69727a;">Metal</div>
            <div class="bin" style="background:#d8a225;color:#201b10;">Papier</div>
            <div class="bin" style="background:#247a4b;">Plastique</div>
            <div class="bin" style="background:#d7634f;">Autre</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_result(result):
    color = CATEGORY_COLORS.get(result["class_name"], "#247a4b")
    confidence = max(0, min(100, result["confidence"]))

    st.markdown(
        f"""
        <div class="result-card">
          <div class="eyebrow">Resultat</div>
          <h2 class="result-label" style="color:{color};">{result["label"]}</h2>
          <div class="confidence-row">
            <span>Confiance du modele</span>
            <strong>{confidence:.2f}%</strong>
          </div>
          <div class="meter"><div class="meter-fill" style="width:{confidence}%;"></div></div>
          <p class="tip">{result["tip"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Top 3")
    for item in result["top_predictions"]:
        st.markdown(
            f"""
            <div class="top-row">
              <span>{item["label"]}</span>
              <strong>{item["confidence"]:.2f}%</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    inject_styles()
    render_hero()

    left, right = st.columns([0.92, 1.08], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<p class="panel-title">Image a analyser</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="panel-copy">Choisissez un fichier local ou collez une URL directe vers une image JPG, PNG ou WebP.</p>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Deposer une image",
            type=sorted(ALLOWED_EXTENSIONS),
            label_visibility="collapsed",
        )
        image_url = st.text_input(
            "URL directe d'une image",
            placeholder="https://exemple.com/photo-bouteille.jpg",
        )
        analyze = st.button("Analyser l'image", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    selected_image_path = None
    error_message = None

    if analyze:
        try:
            if uploaded_file is not None:
                selected_image_path = save_uploaded_file(uploaded_file)
            elif image_url.strip():
                selected_image_path = save_image_from_url(image_url.strip())
            else:
                error_message = "Ajoutez une image ou collez une URL avant l'analyse."
        except requests.RequestException:
            error_message = "Impossible de telecharger cette image. Essayez un autre lien direct."
        except ValueError as exc:
            error_message = str(exc)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<p class="panel-title">Prediction</p>', unsafe_allow_html=True)

        if selected_image_path is not None:
            st.image(str(selected_image_path), use_container_width=True)
            with st.spinner("Analyse du modele en cours..."):
                try:
                    result = predict_image(selected_image_path)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    render_result(result)
        elif error_message:
            st.error(error_message)
            st.markdown('<div class="empty-state">Aucune image analysee</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state">Aucune image analysee</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
