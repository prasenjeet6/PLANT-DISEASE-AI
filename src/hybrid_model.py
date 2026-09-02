import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

DATASET_PATH = "dataset/color"

MOBILENET_WEIGHT = 0.5
EFFICIENTNET_WEIGHT = 0.5

mobilenet_datagen = ImageDataGenerator(
    rescale=1./255
)

efficientnet_datagen = ImageDataGenerator(
    preprocessing_function=efficientnet_preprocess
)

mobilenet_generator = mobilenet_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

efficientnet_generator = efficientnet_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

mobilenet_model = load_model("models/mobilenet.keras")
efficientnet_model = load_model("models/efficientnet.keras")

mobilenet_predictions = mobilenet_model.predict(
    mobilenet_generator,
    verbose=1
)

efficientnet_predictions = efficientnet_model.predict(
    efficientnet_generator,
    verbose=1
)

hybrid_predictions = (
    MOBILENET_WEIGHT * mobilenet_predictions +
    EFFICIENTNET_WEIGHT * efficientnet_predictions
)

y_pred = np.argmax(hybrid_predictions, axis=1)
y_true = mobilenet_generator.classes

class_names = list(mobilenet_generator.class_indices.keys())

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n========== HYBRID MODEL METRICS ==========\n")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print("\n========== CLASSIFICATION REPORT ==========\n")
print(report)

os.makedirs("results/hybrid", exist_ok=True)

with open("results/hybrid/hybrid_metrics.txt", "w") as f:
    f.write(f"MobileNetV2 Weight : {MOBILENET_WEIGHT}\n")
    f.write(f"EfficientNetV2 Weight : {EFFICIENTNET_WEIGHT}\n\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")

with open("results/hybrid/hybrid_classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(20, 20))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=90,
    colorbar=False
)

plt.tight_layout()

plt.savefig(
    "results/hybrid/hybrid_confusion_matrix.png",
    dpi=300
)

plt.show()

print("\n===================================")
print("Hybrid Model Evaluation Completed")
print("===================================")
print("Results saved in results/hybrid/")