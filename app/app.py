import streamlit as st
from PIL import Image
from pathlib import Path
from uuid import uuid4

from predictor import predict_image


ROOT_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_image(uploaded_file):
    extension = Path(uploaded_file.name).suffix.lower()
    image_path = UPLOAD_DIR / f"{uuid4().hex}{extension}"
    image_path.write_bytes(uploaded_file.getvalue())
    return image_path


st.title("RecyclAI - Classification des dechets")
st.write("Charge une image d'un dechet")

uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image chargee")

    if st.button("Classifier l'image"):
        image_path = save_uploaded_image(uploaded_file)
        result = predict_image(image_path)

        st.success(f"Categorie : {result['label'].lower()}")
        st.write(f"Confiance : {result['confidence']:.2f} %")

        st.write("**Top 3 :**")
        for item in result["top_predictions"]:
            st.write(f"- {item['label'].lower()} : {item['confidence']:.2f} %")
