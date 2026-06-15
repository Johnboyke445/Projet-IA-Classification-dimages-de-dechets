from functools import lru_cache
from pathlib import Path
import json
import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "modele_pfe_dechets.keras"
LEGACY_MODEL_PATH = ROOT_DIR / "modele_pfe_dechets.h5"
CLASS_NAMES_PATH = ROOT_DIR / "class_names.json"
IMG_SIZE = (224, 224)

DEFAULT_CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

CLASS_LABELS_FR = {
    "cardboard": "Carton",
    "glass": "Verre",
    "metal": "Metal",
    "paper": "Papier",
    "plastic": "Plastique",
    "trash": "Dechet residuel",
}

SORTING_TIPS_FR = {
    "cardboard": "Aplatir le carton propre et sec, puis le placer avec les emballages recyclables.",
    "glass": "Le verre se depose generalement dans une borne a verre, sans bouchon ni couvercle.",
    "metal": "Les canettes et boites metalliques vides vont avec les emballages recyclables.",
    "paper": "Le papier propre et sec peut etre trie avec les papiers ou emballages recyclables.",
    "plastic": "Les bouteilles et flacons plastiques vides vont avec les emballages recyclables.",
    "trash": "A deposer dans les ordures menageres si l'objet n'est pas recyclable localement.",
}


class CompatibleBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(
        self,
        *args,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)


@lru_cache(maxsize=1)
def load_model():
    custom_objects = {"BatchNormalization": CompatibleBatchNormalization}
    if MODEL_PATH.exists():
        return tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects=custom_objects,
            compile=False,
        )
    if LEGACY_MODEL_PATH.exists():
        return tf.keras.models.load_model(
            LEGACY_MODEL_PATH,
            custom_objects=custom_objects,
            compile=False,
        )
    raise FileNotFoundError(
        f"Aucun modele trouve. Lancez d'abord train_ia.py pour creer {MODEL_PATH.name}."
    )


@lru_cache(maxsize=1)
def load_class_names():
    if CLASS_NAMES_PATH.exists():
        return json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    return DEFAULT_CLASS_NAMES


def predict_image(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    model = load_model()
    class_names = load_class_names()

    image = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)

    predictions = model.predict(image_array, verbose=0)[0]
    best_index = int(np.argmax(predictions))
    best_class = class_names[best_index]

    top_indices = np.argsort(predictions)[::-1][:3]
    top_predictions = [
        {
            "class_name": class_names[int(index)],
            "label": CLASS_LABELS_FR.get(class_names[int(index)], class_names[int(index)]),
            "confidence": round(float(predictions[index]) * 100, 2),
        }
        for index in top_indices
    ]

    return {
        "class_name": best_class,
        "label": CLASS_LABELS_FR.get(best_class, best_class),
        "confidence": round(float(predictions[best_index]) * 100, 2),
        "tip": SORTING_TIPS_FR.get(best_class, "Verifier les consignes de tri de votre commune."),
        "top_predictions": top_predictions,
    }
