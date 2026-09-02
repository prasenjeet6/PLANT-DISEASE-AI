import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = "dataset/color"

mobilenet_datagen = ImageDataGenerator(rescale=1./255)
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

y_true = mobilenet_generator.classes

best_accuracy = 0
best_weight = 0

results = []

for mobilenet_weight in np.arange(0.0, 1.01, 0.1):

    efficientnet_weight = 1.0 - mobilenet_weight

    hybrid_predictions = (
        mobilenet_weight * mobilenet_predictions +
        efficientnet_weight * efficientnet_predictions
    )

    y_pred = np.argmax(hybrid_predictions, axis=1)

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

    results.append(
        (
            mobilenet_weight,
            efficientnet_weight,
            accuracy,
            precision,
            recall,
            f1
        )
    )

    print(
        f"MobileNetV2: {mobilenet_weight:.1f} | "
        f"EfficientNetV2B0: {efficientnet_weight:.1f} | "
        f"Accuracy: {accuracy:.4f}"
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_weight = mobilenet_weight

best_efficientnet_weight = 1.0 - best_weight

best_predictions = (
    best_weight * mobilenet_predictions +
    best_efficientnet_weight * efficientnet_predictions
)

best_y_pred = np.argmax(best_predictions, axis=1)

best_precision = precision_score(
    y_true,
    best_y_pred,
    average="weighted",
    zero_division=0
)

best_recall = recall_score(
    y_true,
    best_y_pred,
    average="weighted",
    zero_division=0
)

best_f1 = f1_score(
    y_true,
    best_y_pred,
    average="weighted",
    zero_division=0
)

os.makedirs("results/optimization", exist_ok=True)

with open("results/optimization/optimized_hybrid_results.txt", "w") as f:

    f.write("OPTIMIZED HYBRID MODEL RESULTS\n")
    f.write("==============================\n\n")

    f.write(
        f"Best MobileNetV2 Weight : {best_weight:.1f}\n"
    )

    f.write(
        f"Best EfficientNetV2B0 Weight : "
        f"{best_efficientnet_weight:.1f}\n\n"
    )

    f.write(f"Accuracy : {best_accuracy:.4f}\n")
    f.write(f"Precision: {best_precision:.4f}\n")
    f.write(f"Recall   : {best_recall:.4f}\n")
    f.write(f"F1 Score : {best_f1:.4f}\n\n")

    f.write("ALL WEIGHT COMBINATIONS\n")
    f.write("=======================\n")

    for result in results:
        f.write(
            f"MobileNetV2={result[0]:.1f}, "
            f"EfficientNetV2B0={result[1]:.1f}, "
            f"Accuracy={result[2]:.4f}, "
            f"Precision={result[3]:.4f}, "
            f"Recall={result[4]:.4f}, "
            f"F1={result[5]:.4f}\n"
        )

print("\n===================================")
print("HYBRID WEIGHT OPTIMIZATION COMPLETED")
print("===================================")

print(
    f"Best MobileNetV2 Weight: {best_weight:.1f}"
)

print(
    f"Best EfficientNetV2B0 Weight: "
    f"{best_efficientnet_weight:.1f}"
)

print(f"Best Accuracy : {best_accuracy:.4f}")
print(f"Best Precision: {best_precision:.4f}")
print(f"Best Recall   : {best_recall:.4f}")
print(f"Best F1 Score : {best_f1:.4f}")

print(
    "\nResults saved in "
    "results/optimization/"
)