from pathlib import Path
import json

import tensorflow as tf
from tensorflow.keras import layers, models


# --- 1. CONFIGURATION ---
BASE_PATH = Path("trashnet/dataset_split")
TRAIN_DIR = BASE_PATH / "train"
VAL_DIR = BASE_PATH / "val"
TEST_DIR = BASE_PATH / "test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

EPOCHS_HEAD = 10
EPOCHS_FINE_TUNE = 8
FINE_TUNE_LAST_LAYERS = 30

MODEL_PATH = Path("modele_pfe_dechets.keras")
CLASS_NAMES_PATH = Path("class_names.json")


def build_dataset(directory, shuffle=False):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


# --- 2. CHARGEMENT DES DONNEES ---
train_ds = build_dataset(TRAIN_DIR, shuffle=True)
val_ds = build_dataset(VAL_DIR)
test_ds = build_dataset(TEST_DIR)

class_names = train_ds.class_names
CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
print(f"Categories detectees : {class_names}")

autotune = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(autotune)
val_ds = val_ds.cache().prefetch(autotune)
test_ds = test_ds.cache().prefetch(autotune)


# --- 3. MODELE ---
data_augmentation = models.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.12),
        layers.RandomContrast(0.15),
    ],
    name="data_augmentation",
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = models.Model(inputs, outputs)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
    ),
]


# --- 4. ENTRAINEMENT DE LA TETE ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

print("\nPhase 1/2 : entrainement de la tete du modele...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks,
)


# --- 5. FINE-TUNING ---
base_model.trainable = True
for layer in base_model.layers[:-FINE_TUNE_LAST_LAYERS]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

print("\nPhase 2/2 : fine-tuning des dernieres couches MobileNetV2...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD + EPOCHS_FINE_TUNE,
    initial_epoch=EPOCHS_HEAD,
    callbacks=callbacks,
)


# --- 6. EVALUATION ET SAUVEGARDE ---
best_model = tf.keras.models.load_model(MODEL_PATH)

print("\n--- Evaluation finale sur le dossier TEST ---")
test_loss, test_accuracy = best_model.evaluate(test_ds)
print(f"Accuracy test : {test_accuracy:.4f}")
print(f"Loss test : {test_loss:.4f}")

best_model.save(MODEL_PATH)
print(f"Modele sauvegarde : {MODEL_PATH}")
print(f"Classes sauvegardees : {CLASS_NAMES_PATH}")
