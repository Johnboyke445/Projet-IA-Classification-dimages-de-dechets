import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "modele_pfe_dechets.h5"
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def classifier(image, model):
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    classification = model.predict(img_array, verbose=0)[0]
    best_index = int(np.argmax(classification))
    return classification, best_index


st.title("♻️ RecyclAI - Classification des déchets")
st.write("Charge une image d'un déchet 👇")

uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image chargée")

    if st.button("Classifier l'image"):
        model = load_model()
        classification, best_index = classifier(image, model)

        st.success(f"Catégorie : {CLASS_NAMES[best_index]}")
        st.write(f"Confiance : {classification[best_index] * 100:.2f}%")

        st.write("**Top 3 :**")
        top_indices = np.argsort(classification)[::-1][:3]
        for i in top_indices:
            st.write(f"- {CLASS_NAMES[i]} : {classification[i] * 100:.2f}%")
