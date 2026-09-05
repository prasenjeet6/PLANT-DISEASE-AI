import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, roc_auc_score

dataset_path = "dataset/color"
model_path = "models/vit.keras"

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

model = load_model(model_path)

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

print("\nViT Validation Loss:", validation_loss)
print("ViT Validation Accuracy:", validation_accuracy)
print("ViT Weighted Precision:", precision)
print("ViT Weighted Recall:", recall)
print("ViT Weighted F1:", f1)
print("ViT Weighted ROC-AUC:", roc_auc)

print("\nClassification Report:\n")
print(report)

with open(
    "results/reports/vit_classification_report.txt",
    "w"
) as file:
    file.write(report)
    file.write("\nViT Validation Loss: " + str(validation_loss))
    file.write("\nViT Validation Accuracy: " + str(validation_accuracy))
    file.write("\nViT Weighted Precision: " + str(precision))
    file.write("\nViT Weighted Recall: " + str(recall))
    file.write("\nViT Weighted F1: " + str(f1))
    file.write("\nViT Weighted ROC-AUC: " + str(roc_auc))

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

plt.figure(figsize=(20, 18))
plt.imshow(cm)
plt.title("ViT Confusion Matrix")
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
    "results/confusion_matrices/vit_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nViT evaluation completed successfully.")
print("Classification report saved.")
print("Confusion matrix saved.")