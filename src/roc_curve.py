import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

DATASET_PATH = "dataset/color"

MOBILENET_PATH = "models/mobilenet.keras"
EFFICIENTNET_PATH = "models/efficientnet.keras"

RESULT_PATH = "results/graphs"

os.makedirs(RESULT_PATH, exist_ok=True)

MOBILE_WEIGHT = 0.414
EFFICIENT_WEIGHT = 0.586

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
num_classes = len(class_names)

print("Number of classes:", num_classes)
print("Validation images:", mobilenet_generator.samples)

mobilenet_model = load_model(MOBILENET_PATH)
efficientnet_model = load_model(EFFICIENTNET_PATH)

mobilenet_generator.reset()

mobile_probabilities = mobilenet_model.predict(
    mobilenet_generator,
    verbose=1
)

efficientnet_generator.reset()

efficient_probabilities = efficientnet_model.predict(
    efficientnet_generator,
    verbose=1
)

y_true = mobilenet_generator.classes

y_true_binary = label_binarize(
    y_true,
    classes=np.arange(num_classes)
)

hybrid_probabilities = (
    0.5 * mobile_probabilities
    + 0.5 * efficient_probabilities
)

pso_probabilities = (
    MOBILE_WEIGHT * mobile_probabilities
    + EFFICIENT_WEIGHT * efficient_probabilities
)

def calculate_micro_roc(y_true_binary, probabilities):
    fpr, tpr, _ = roc_curve(
        y_true_binary.ravel(),
        probabilities.ravel()
    )

    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

mobile_fpr, mobile_tpr, mobile_auc = calculate_micro_roc(
    y_true_binary,
    mobile_probabilities
)

efficient_fpr, efficient_tpr, efficient_auc = calculate_micro_roc(
    y_true_binary,
    efficient_probabilities
)

hybrid_fpr, hybrid_tpr, hybrid_auc = calculate_micro_roc(
    y_true_binary,
    hybrid_probabilities
)

pso_fpr, pso_tpr, pso_auc = calculate_micro_roc(
    y_true_binary,
    pso_probabilities
)

print()
print("ROC-AUC RESULTS")
print("=" * 50)

print(f"MobileNetV2 AUC          : {mobile_auc:.4f}")
print(f"EfficientNetV2B0 AUC     : {efficient_auc:.4f}")
print(f"50:50 Hybrid AUC         : {hybrid_auc:.4f}")
print(f"PSO Optimized Hybrid AUC : {pso_auc:.4f}")

plt.figure(figsize=(10, 8))

plt.plot(
    mobile_fpr,
    mobile_tpr,
    label=f"MobileNetV2 (AUC = {mobile_auc:.4f})"
)

plt.plot(
    efficient_fpr,
    efficient_tpr,
    label=f"EfficientNetV2B0 (AUC = {efficient_auc:.4f})"
)

plt.plot(
    hybrid_fpr,
    hybrid_tpr,
    label=f"50:50 Hybrid (AUC = {hybrid_auc:.4f})"
)

plt.plot(
    pso_fpr,
    pso_tpr,
    label=f"PSO Hybrid (AUC = {pso_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

output_path = os.path.join(
    RESULT_PATH,
    "roc_curve_comparison.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

auc_path = os.path.join(
    RESULT_PATH,
    "roc_auc_results.txt"
)

with open(
    auc_path,
    "w",
    encoding="utf-8"
) as f:
    f.write("ROC-AUC RESULTS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"MobileNetV2 AUC: {mobile_auc:.4f}\n")
    f.write(f"EfficientNetV2B0 AUC: {efficient_auc:.4f}\n")
    f.write(f"50:50 Hybrid AUC: {hybrid_auc:.4f}\n")
    f.write(f"PSO Optimized Hybrid AUC: {pso_auc:.4f}\n")

print()
print("ROC curve completed successfully.")
print("Graph saved to:", output_path)
print("AUC results saved to:", auc_path)