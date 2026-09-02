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

MOBILENET_PATH = "models/mobilenet.keras"
EFFICIENTNET_PATH = "models/efficientnet.keras"

RESULT_PATH = "results/optimization"
GRAPH_PATH = "results/graphs"

os.makedirs(RESULT_PATH, exist_ok=True)
os.makedirs(GRAPH_PATH, exist_ok=True)

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

print()
print("Validation images:", mobilenet_generator.samples)
print("Number of classes:", len(mobilenet_generator.class_indices))

mobilenet_model = load_model(MOBILENET_PATH)
efficientnet_model = load_model(EFFICIENTNET_PATH)

print()
print("Generating MobileNetV2 predictions...")

mobilenet_generator.reset()

mobilenet_predictions = mobilenet_model.predict(
    mobilenet_generator,
    verbose=1
)

print()
print("Generating EfficientNetV2B0 predictions...")

efficientnet_generator.reset()

efficientnet_predictions = efficientnet_model.predict(
    efficientnet_generator,
    verbose=1
)

y_true = mobilenet_generator.classes

class_names = list(
    mobilenet_generator.class_indices.keys()
)

def evaluate_weight(weight):

    efficientnet_weight = 1.0 - weight

    predictions = (
        weight * mobilenet_predictions
        + efficientnet_weight * efficientnet_predictions
    )

    y_pred = np.argmax(
        predictions,
        axis=1
    )

    return accuracy_score(
        y_true,
        y_pred
    )

particles = 5
iterations = 6

np.random.seed(42)

positions = np.random.uniform(
    0.0,
    1.0,
    particles
)

velocities = np.random.uniform(
    -0.1,
    0.1,
    particles
)

personal_best_positions = positions.copy()

personal_best_scores = np.array(
    [
        evaluate_weight(position)
        for position in positions
    ]
)

best_index = np.argmax(
    personal_best_scores
)

global_best_position = (
    personal_best_positions[best_index]
)

global_best_score = (
    personal_best_scores[best_index]
)

w = 0.7
c1 = 1.5
c2 = 1.5

convergence_history = [
    global_best_score
]

weight_history = [
    global_best_position
]

print()
print("=" * 70)
print("PSO OPTIMIZATION")
print("=" * 70)

for iteration in range(iterations):

    for i in range(particles):

        r1 = np.random.random()
        r2 = np.random.random()

        velocities[i] = (
            w * velocities[i]
            + c1 * r1
            * (
                personal_best_positions[i]
                - positions[i]
            )
            + c2 * r2
            * (
                global_best_position
                - positions[i]
            )
        )

        positions[i] += velocities[i]

        positions[i] = np.clip(
            positions[i],
            0.0,
            1.0
        )

        score = evaluate_weight(
            positions[i]
        )

        if score > personal_best_scores[i]:

            personal_best_scores[i] = score

            personal_best_positions[i] = (
                positions[i]
            )

        if score > global_best_score:

            global_best_score = score

            global_best_position = (
                positions[i]
            )

    convergence_history.append(
        global_best_score
    )

    weight_history.append(
        global_best_position
    )

    print(
        f"Iteration {iteration + 1}/{iterations} | "
        f"Best Weight: {global_best_position:.4f} | "
        f"Accuracy: {global_best_score:.4f}"
    )

best_weight = global_best_position

best_efficientnet_weight = (
    1.0 - best_weight
)

best_predictions = (
    best_weight * mobilenet_predictions
    + best_efficientnet_weight
    * efficientnet_predictions
)

best_y_pred = np.argmax(
    best_predictions,
    axis=1
)

best_accuracy = accuracy_score(
    y_true,
    best_y_pred
)

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

hybrid_50_predictions = (
    0.5 * mobilenet_predictions
    + 0.5 * efficientnet_predictions
)

hybrid_50_y_pred = np.argmax(
    hybrid_50_predictions,
    axis=1
)

hybrid_50_accuracy = accuracy_score(
    y_true,
    hybrid_50_y_pred
)

hybrid_50_precision = precision_score(
    y_true,
    hybrid_50_y_pred,
    average="weighted",
    zero_division=0
)

hybrid_50_recall = recall_score(
    y_true,
    hybrid_50_y_pred,
    average="weighted",
    zero_division=0
)

hybrid_50_f1 = f1_score(
    y_true,
    hybrid_50_y_pred,
    average="weighted",
    zero_division=0
)

report = classification_report(
    y_true,
    best_y_pred,
    target_names=class_names,
    zero_division=0
)

cm = confusion_matrix(
    y_true,
    best_y_pred
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

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    range(len(convergence_history)),
    convergence_history,
    marker="o",
    label="Best Accuracy"
)

plt.xlabel(
    "PSO Iteration"
)

plt.ylabel(
    "Best Accuracy"
)

plt.title(
    "PSO Optimization Convergence"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_PATH,
        "pso_convergence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    range(len(weight_history)),
    weight_history,
    marker="o",
    label="Best MobileNetV2 Weight"
)

plt.xlabel(
    "PSO Iteration"
)

plt.ylabel(
    "MobileNetV2 Weight"
)

plt.title(
    "PSO Weight Optimization"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_PATH,
        "pso_weight_optimization.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

with open(
    os.path.join(
        RESULT_PATH,
        "pso_hybrid_classification_report.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

with open(
    os.path.join(
        RESULT_PATH,
        "pso_hybrid_results.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PSO OPTIMIZED HYBRID MODEL RESULTS\n"
    )

    f.write(
        "=================================\n\n"
    )

    f.write(
        f"Validation Images : "
        f"{len(y_true)}\n\n"
    )

    f.write(
        f"Best MobileNetV2 Weight : "
        f"{best_weight:.4f}\n"
    )

    f.write(
        f"Best EfficientNetV2B0 Weight : "
        f"{best_efficientnet_weight:.4f}\n\n"
    )

    f.write(
        f"Accuracy : {best_accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {best_precision:.4f}\n"
    )

    f.write(
        f"Recall   : {best_recall:.4f}\n"
    )

    f.write(
        f"F1 Score : {best_f1:.4f}\n\n"
    )

    f.write(
        "50:50 HYBRID RESULTS\n"
    )

    f.write(
        "====================\n\n"
    )

    f.write(
        f"Accuracy : "
        f"{hybrid_50_accuracy:.4f}\n"
    )

    f.write(
        f"Precision: "
        f"{hybrid_50_precision:.4f}\n"
    )

    f.write(
        f"Recall   : "
        f"{hybrid_50_recall:.4f}\n"
    )

    f.write(
        f"F1 Score : "
        f"{hybrid_50_f1:.4f}\n"
    )

print()
print("=" * 70)
print("PSO OPTIMIZATION COMPLETED")
print("=" * 70)

print()
print(
    f"Validation Images : {len(y_true)}"
)

print(
    f"Best MobileNetV2 Weight : "
    f"{best_weight:.4f}"
)

print(
    f"Best EfficientNetV2B0 Weight : "
    f"{best_efficientnet_weight:.4f}"
)

print()
print(
    f"PSO Accuracy : "
    f"{best_accuracy:.4f}"
)

print(
    f"PSO Precision: "
    f"{best_precision:.4f}"
)

print(
    f"PSO Recall   : "
    f"{best_recall:.4f}"
)

print(
    f"PSO F1 Score : "
    f"{best_f1:.4f}"
)

print()
print(
    f"50:50 Hybrid Accuracy : "
    f"{hybrid_50_accuracy:.4f}"
)

print(
    f"50:50 Hybrid F1 Score : "
    f"{hybrid_50_f1:.4f}"
)

print()
print(
    "Results saved in:"
)

print(
    "results/optimization/"
)

print(
    "results/graphs/"
)