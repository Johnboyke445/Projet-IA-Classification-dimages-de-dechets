from pathlib import Path
import json
import sys

import numpy as np
import tensorflow as tf


MODEL_PATH = Path("modele_pfe_dechets.keras")
LEGACY_MODEL_PATH = Path("modele_pfe_dechets.h5")
CLASS_NAMES_PATH = Path("class_names.json")
DEFAULT_IMAGE = Path("trashnet/dataset_split/test/plastic/plastic410.jpg")


def load_model():
    if MODEL_PATH.exists():
        return tf.keras.models.load_model(MODEL_PATH)
    if LEGACY_MODEL_PATH.exists():
        return tf.keras.models.load_model(LEGACY_MODEL_PATH)
    raise FileNotFoundError(
        f"Aucun modele trouve. Lance d'abord train_ia.py pour creer {MODEL_PATH}."
    )


def load_class_names():
    if CLASS_NAMES_PATH.exists():
        return json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    return ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def predire_dechet(chemin_image):
    image_path = Path(chemin_image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array, verbose=0)[0]
    best_index = int(np.argmax(predictions))
    confidence = float(predictions[best_index])

    print(f"\nResultat pour {image_path} :")
    print(f"Classe predite : {class_names[best_index]}")
    print(f"Confiance : {confidence * 100:.2f}%")

    top_indices = np.argsort(predictions)[::-1][:3]
    print("\nTop 3 :")
    for index in top_indices:
        print(f"- {class_names[int(index)]}: {float(predictions[index]) * 100:.2f}%")


model = load_model()
class_names = load_class_names()

image_to_test = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
predire_dechet(image_to_test)
