import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = "dataset/color"

MOBILE_WEIGHT = 0.3745
EFFICIENT_WEIGHT = 0.6255

RESULT_PATH = "results/optimization"

os.makedirs(RESULT_PATH, exist_ok=True)

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
    shuffle=False,
    seed=42
)

efficientnet_generator = efficientnet_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42
)

class_names = list(
    mobilenet_generator.class_indices.keys()
)

mobilenet_model = load_model(
    "models/mobilenet.keras"
)

efficientnet_model = load_model(
    "models/efficientnet.keras"
)

mobilenet_generator.reset()

mobilenet_predictions = mobilenet_model.predict(
    mobilenet_generator,
    verbose=1
)

efficientnet_generator.reset()

efficientnet_predictions = efficientnet_model.predict(
    efficientnet_generator,
    verbose=1
)

y_true = mobilenet_generator.classes

hybrid_predictions = (
    MOBILE_WEIGHT * mobilenet_predictions
    + EFFICIENT_WEIGHT * efficientnet_predictions
)

y_pred = np.argmax(
    hybrid_predictions,
    axis=1
)

accuracy = accuracy_score(
    y_true,
    y_pred
)

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

print()
print("=" * 70)
print("PSO HYBRID EVALUATION")
print("=" * 70)

print()
print(
    f"MobileNetV2 Weight : {MOBILE_WEIGHT:.4f}"
)

print(
    f"EfficientNetV2B0 Weight : {EFFICIENT_WEIGHT:.4f}"
)

print()
print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print()
print(report)

with open(
    os.path.join(
        RESULT_PATH,
        "pso_hybrid_classification_report.txt"
    ),
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(
    figsize=(18, 16)
)

plt.imshow(
    cm,
    cmap="Blues"
)

plt.title(
    "PSO Optimized Hybrid Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

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
        "pso_hybrid_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

with open(
    os.path.join(
        RESULT_PATH,
        "pso_hybrid_final_metrics.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PSO OPTIMIZED HYBRID MODEL\n"
    )

    f.write(
        "===========================\n\n"
    )

    f.write(
        f"Validation Images: "
        f"{len(y_true)}\n\n"
    )

    f.write(
        f"MobileNetV2 Weight: "
        f"{MOBILE_WEIGHT:.4f}\n"
    )

    f.write(
        f"EfficientNetV2B0 Weight: "
        f"{EFFICIENT_WEIGHT:.4f}\n\n"
    )

    f.write(
        f"Accuracy: "
        f"{accuracy:.4f}\n"
    )

    f.write(
        f"Precision: "
        f"{precision:.4f}\n"
    )

    f.write(
        f"Recall: "
        f"{recall:.4f}\n"
    )

    f.write(
        f"F1 Score: "
        f"{f1:.4f}\n"
    )

print()
print("PSO hybrid evaluation completed successfully.")
print(
    "Results saved to results/optimization/"
)