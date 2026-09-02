import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

DATASET_PATH = "dataset/color"

MOBILENET_PATH = "models/mobilenet.keras"
EFFICIENTNET_PATH = "models/efficientnet.keras"

RESULT_PATH = "results/evaluation"

os.makedirs(RESULT_PATH, exist_ok=True)

print("=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)

mobilenet_datagen = ImageDataGenerator(
    preprocessing_function=mobilenet_preprocess,
    validation_split=0.2
)

efficientnet_datagen = ImageDataGenerator(
    preprocessing_function=efficientnet_preprocess,
    validation_split=0.2
)

mobilenet_generator = mobilenet_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

efficientnet_generator = efficientnet_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

class_names = list(mobilenet_generator.class_indices.keys())

print()
print("Number of classes:", len(class_names))
print("Validation images:", mobilenet_generator.samples)

print()
print("Loading MobileNetV2...")
mobilenet_model = load_model(MOBILENET_PATH)
print("MobileNetV2 loaded.")

print()
print("Loading EfficientNetV2B0...")
efficientnet_model = load_model(EFFICIENTNET_PATH)
print("EfficientNetV2B0 loaded.")

print()
print("=" * 70)
print("MOBILENETV2 EVALUATION")
print("=" * 70)

mobilenet_generator.reset()

mobile_probabilities = mobilenet_model.predict(
    mobilenet_generator,
    verbose=1
)

mobile_predictions = np.argmax(
    mobile_probabilities,
    axis=1
)

mobile_true = mobilenet_generator.classes

mobile_accuracy = accuracy_score(
    mobile_true,
    mobile_predictions
)

mobile_precision = precision_score(
    mobile_true,
    mobile_predictions,
    average="weighted",
    zero_division=0
)

mobile_recall = recall_score(
    mobile_true,
    mobile_predictions,
    average="weighted",
    zero_division=0
)

mobile_f1 = f1_score(
    mobile_true,
    mobile_predictions,
    average="weighted",
    zero_division=0
)

print()
print("MobileNetV2 Accuracy :", f"{mobile_accuracy * 100:.2f}%")
print("MobileNetV2 Precision:", f"{mobile_precision * 100:.2f}%")
print("MobileNetV2 Recall   :", f"{mobile_recall * 100:.2f}%")
print("MobileNetV2 F1 Score :", f"{mobile_f1 * 100:.2f}%")

mobile_report = classification_report(
    mobile_true,
    mobile_predictions,
    target_names=class_names,
    zero_division=0
)

with open(
    os.path.join(RESULT_PATH, "mobilenet_classification_report.txt"),
    "w",
    encoding="utf-8"
) as f:
    f.write(mobile_report)

mobile_cm = confusion_matrix(
    mobile_true,
    mobile_predictions
)

plt.figure(figsize=(18, 16))
plt.imshow(mobile_cm)
plt.title("MobileNetV2 Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.colorbar()
plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=90,
    fontsize=7
)
plt.yticks(
    range(len(class_names)),
    class_names,
    fontsize=7
)
plt.tight_layout()
plt.savefig(
    os.path.join(
        RESULT_PATH,
        "mobilenet_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print()
print("=" * 70)
print("EFFICIENTNETV2B0 EVALUATION")
print("=" * 70)

efficientnet_generator.reset()

efficient_probabilities = efficientnet_model.predict(
    efficientnet_generator,
    verbose=1
)

efficient_predictions = np.argmax(
    efficient_probabilities,
    axis=1
)

efficient_true = efficientnet_generator.classes

efficient_accuracy = accuracy_score(
    efficient_true,
    efficient_predictions
)

efficient_precision = precision_score(
    efficient_true,
    efficient_predictions,
    average="weighted",
    zero_division=0
)

efficient_recall = recall_score(
    efficient_true,
    efficient_predictions,
    average="weighted",
    zero_division=0
)

efficient_f1 = f1_score(
    efficient_true,
    efficient_predictions,
    average="weighted",
    zero_division=0
)

print()
print(
    "EfficientNetV2B0 Accuracy :",
    f"{efficient_accuracy * 100:.2f}%"
)

print(
    "EfficientNetV2B0 Precision:",
    f"{efficient_precision * 100:.2f}%"
)

print(
    "EfficientNetV2B0 Recall   :",
    f"{efficient_recall * 100:.2f}%"
)

print(
    "EfficientNetV2B0 F1 Score :",
    f"{efficient_f1 * 100:.2f}%"
)

efficient_report = classification_report(
    efficient_true,
    efficient_predictions,
    target_names=class_names,
    zero_division=0
)

with open(
    os.path.join(
        RESULT_PATH,
        "efficientnet_classification_report.txt"
    ),
    "w",
    encoding="utf-8"
) as f:
    f.write(efficient_report)

efficient_cm = confusion_matrix(
    efficient_true,
    efficient_predictions
)

plt.figure(figsize=(18, 16))
plt.imshow(efficient_cm)
plt.title("EfficientNetV2B0 Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.colorbar()
plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=90,
    fontsize=7
)
plt.yticks(
    range(len(class_names)),
    class_names,
    fontsize=7
)
plt.tight_layout()
plt.savefig(
    os.path.join(
        RESULT_PATH,
        "efficientnet_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print()
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print()
print(
    f"{'Metric':<20}"
    f"{'MobileNetV2':<20}"
    f"{'EfficientNetV2B0':<20}"
)

print("-" * 60)

print(
    f"{'Accuracy':<20}"
    f"{mobile_accuracy * 100:<20.2f}"
    f"{efficient_accuracy * 100:<20.2f}"
)

print(
    f"{'Precision':<20}"
    f"{mobile_precision * 100:<20.2f}"
    f"{efficient_precision * 100:<20.2f}"
)

print(
    f"{'Recall':<20}"
    f"{mobile_recall * 100:<20.2f}"
    f"{efficient_recall * 100:<20.2f}"
)

print(
    f"{'F1 Score':<20}"
    f"{mobile_f1 * 100:<20.2f}"
    f"{efficient_f1 * 100:<20.2f}"
)

if efficient_f1 > mobile_f1:
    selected_model = "EfficientNetV2B0"
    selected_score = efficient_f1
else:
    selected_model = "MobileNetV2"
    selected_score = mobile_f1

print()
print("=" * 70)
print("SELECTED MODEL")
print("=" * 70)

print("Model:", selected_model)
print("F1 Score:", f"{selected_score * 100:.2f}%")

summary_path = os.path.join(
    RESULT_PATH,
    "final_model_comparison.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:
    f.write("FINAL MODEL COMPARISON\n")
    f.write("=" * 70 + "\n\n")
    f.write(
        f"MobileNetV2 Accuracy: "
        f"{mobile_accuracy * 100:.2f}%\n"
    )
    f.write(
        f"MobileNetV2 Precision: "
        f"{mobile_precision * 100:.2f}%\n"
    )
    f.write(
        f"MobileNetV2 Recall: "
        f"{mobile_recall * 100:.2f}%\n"
    )
    f.write(
        f"MobileNetV2 F1 Score: "
        f"{mobile_f1 * 100:.2f}%\n\n"
    )

    f.write(
        f"EfficientNetV2B0 Accuracy: "
        f"{efficient_accuracy * 100:.2f}%\n"
    )
    f.write(
        f"EfficientNetV2B0 Precision: "
        f"{efficient_precision * 100:.2f}%\n"
    )
    f.write(
        f"EfficientNetV2B0 Recall: "
        f"{efficient_recall * 100:.2f}%\n"
    )
    f.write(
        f"EfficientNetV2B0 F1 Score: "
        f"{efficient_f1 * 100:.2f}%\n\n"
    )

    f.write(
        f"Selected Model: {selected_model}\n"
    )

print()
print("Evaluation completed successfully.")
print("Results saved to:", RESULT_PATH)
print("=" * 70)