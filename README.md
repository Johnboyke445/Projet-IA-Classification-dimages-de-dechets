# Tri IA - Classification d'images de dechets

Application Python/TensorFlow avec interface Streamlit pour classifier une image en six categories:
carton, verre, metal, papier, plastique ou dechet residuel.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Lancer l'application Streamlit

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Ouvrir ensuite: http://localhost:8501

L'interface accepte deux sources:
- une image locale telechargee depuis votre ordinateur;
- l'URL directe d'une image en ligne, par exemple une photo JPG/PNG/WebP trouvee via Google Images.

## Ancienne interface Flask

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Ouvrir ensuite: http://127.0.0.1:5000

## Tester une image en ligne de commande

```powershell
.\.venv\Scripts\Activate.ps1
python test_image.py trashnet/dataset_split/test/plastic/plastic410.jpg
```

Sans argument, le script utilise cette image plastique de test par defaut.

## Reentrainer le modele

```powershell
.\.venv\Scripts\Activate.ps1
python train_ia.py
```

Le script sauvegarde `modele_pfe_dechets.keras` et `class_names.json`.
