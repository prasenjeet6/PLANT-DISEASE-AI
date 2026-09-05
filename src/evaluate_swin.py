import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, roc_auc_score

original_compute_output_shape = Lambda.compute_output_shape

def patched_compute_output_shape(self, input_shape):
    if isinstance(input_shape, tuple):
        if len(input_shape) == 4 and input_shape[1:] == (56, 56, 32):
            return (None, 49, 32)
        if len(input_shape) == 3 and input_shape[1:] == (49, 32):
            return (None, 56, 56, 32)
    return original_compute_output_shape(self, input_shape)

Lambda.compute_output_shape = patched_compute_output_shape

def window_partition(x):
    window_size = 7
    shape = tf.shape(x)
    batch_size = shape[0]

    x = tf.reshape(
        x,
        [
            batch_size,
            8,
            7,
            8,
            7,
            32
        ]
    )

    x = tf.transpose(
        x,
        [0, 1, 3, 2, 4, 5]
    )

    return tf.reshape(
        x,
        [-1, 49, 32]
    )

def window_reverse(x):
    shape = tf.shape(x)
    batch_size = shape[0] // 64

    x = tf.reshape(
        x,
        [
            batch_size,
            8,
            8,
            7,
            7,
            32
        ]
    )

    x = tf.transpose(
        x,
        [0, 1, 3, 2, 4, 5]
    )

    return tf.reshape(
        x,
        [batch_size, 56, 56, 32]
    )

dataset_path = "dataset/color"
model_path = "models/swin_transformer.keras"

os.makedirs("results/reports", exist_ok=True)
os.makedirs("results/confusion_matrices", exist_ok=True)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

validation_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

class_names = list(validation_data.class_indices.keys())
true_classes = validation_data.classes

model = load_model(
    model_path,
    custom_objects={
        "window_partition": window_partition,
        "window_reverse": window_reverse
    },
    safe_mode=False,
    compile=False
)

for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Lambda):
        function = getattr(layer, "function", None)
        if function is not None and hasattr(function, "__globals__"):
            function.__globals__["tf"] = tf
            function.__globals__["np"] = np

print("\nSwin Transformer model loaded successfully.")

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

results = model.evaluate(
    validation_data,
    verbose=1
)

validation_loss = results[0]
validation_accuracy = results[1]

validation_data.reset()

predictions = model.predict(
    validation_data,
    verbose=1
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

precision, recall, f1, support = precision_recall_fscore_support(
    true_classes,
    predicted_classes,
    average="weighted",
    zero_division=0
)

true_one_hot = tf.keras.utils.to_categorical(
    true_classes,
    num_classes=len(class_names)
)

roc_auc = roc_auc_score(
    true_one_hot,
    predictions,
    multi_class="ovr",
    average="weighted"
)

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\nSwin Transformer Validation Loss:", validation_loss)
print("Swin Transformer Validation Accuracy:", validation_accuracy)
print("Swin Transformer Weighted Precision:", precision)
print("Swin Transformer Weighted Recall:", recall)
print("Swin Transformer Weighted F1:", f1)
print("Swin Transformer Weighted ROC-AUC:", roc_auc)

print("\nClassification Report:\n")
print(report)

with open(
    "results/reports/swin_transformer_classification_report.txt",
    "w"
) as file:
    file.write(report)
    file.write("\nSwin Transformer Validation Loss: " + str(validation_loss))
    file.write("\nSwin Transformer Validation Accuracy: " + str(validation_accuracy))
    file.write("\nSwin Transformer Weighted Precision: " + str(precision))
    file.write("\nSwin Transformer Weighted Recall: " + str(recall))
    file.write("\nSwin Transformer Weighted F1: " + str(f1))
    file.write("\nSwin Transformer Weighted ROC-AUC: " + str(roc_auc))

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

plt.figure(figsize=(20, 18))
plt.imshow(cm)
plt.title("Swin Transformer Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.colorbar()

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=90,
    fontsize=6
)

plt.yticks(
    range(len(class_names)),
    class_names,
    fontsize=6
)

plt.tight_layout()

plt.savefig(
    "results/confusion_matrices/swin_transformer_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSwin Transformer evaluation completed successfully.")
print("Classification report saved.")
print("Confusion matrix saved.")