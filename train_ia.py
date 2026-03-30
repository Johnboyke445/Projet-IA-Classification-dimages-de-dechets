import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION DES CHEMINS ---
BASE_PATH = "trashnet/dataset_split"
TRAIN_DIR = f"{BASE_PATH}/train"
VAL_DIR = f"{BASE_PATH}/val"
TEST_DIR = f"{BASE_PATH}/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- 2. CHARGEMENT DES DONNÉES ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"✅ Catégories détectées : {class_names}")

# --- 3. MODÈLE (TRANSFER LEARNING AVEC MOBILENETV2) ---
# On prend le modèle de Google (pré-entraîné sur 1 million d'images)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # On gèle les couches de Google

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2), # Pour éviter que l'IA n'apprenne par coeur
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# --- 4. ENTRAÎNEMENT ---
print("\n🚀 Lancement de l'entraînement haute précision...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# --- 5. ÉVALUATION ET SAUVEGARDE ---
print("\n--- Évaluation finale sur le dossier TEST ---")
model.evaluate(test_ds)

model.save('modele_pfe_dechets.h5')
print("✅ Nouveau modèle ultra-précis sauvegardé !")