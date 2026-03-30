import tensorflow as tf
import numpy as np
import sys

# 1. Charger le cerveau de l'IA  
model = tf.keras.models.load_model('modele_pfe_dechets.h5')
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 2. FONCTION POUR PRÉDIRE
def predire_dechet(chemin_image):
    # Charger et redimensionner l'image comme pendant l'entraînement
    img = tf.keras.utils.load_img(chemin_image, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Créer un paquet d'une image

    # Faire la prédiction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(f"\n🔍 Résultat pour {chemin_image} :")
    print(f"C'est du **{class_names[np.argmax(score)]}** (Confiance : {100 * np.max(score):.2f}%)")

# 3. TESTER UNE IMAGE
predire_dechet('trashnet/dataset_split/test/plastic/plastic410.jpg')
