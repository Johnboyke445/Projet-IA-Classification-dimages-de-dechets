import sys
from pathlib import Path

from predictor import predict_image


DEFAULT_IMAGE = Path("trashnet/dataset_split/test/plastic/plastic410.jpg")


def predire_dechet(chemin_image):
    result = predict_image(chemin_image)

    print(f"\nResultat pour {Path(chemin_image)} :")
    print(f"Classe predite : {result['label']} ({result['class_name']})")
    print(f"Confiance : {result['confidence']:.2f}%")
    print(f"Conseil : {result['tip']}")

    print("\nTop 3 :")
    for item in result["top_predictions"]:
        print(f"- {item['label']}: {item['confidence']:.2f}%")


if __name__ == "__main__":
    image_to_test = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    predire_dechet(image_to_test)
