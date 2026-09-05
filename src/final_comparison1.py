import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results/graphs", exist_ok=True)

models = [
    "MobileNetV2",
    "EfficientNetV2B0",
    "ResNet50",
    "ViT",
    "Swin Transformer"
]

accuracy = [0.9751, 0.9949, 0.8092, 0.7931, 0.7786]
precision = [0.9753, 0.9949, 0.8244, 0.7891, 0.7708]
recall = [0.9751, 0.9949, 0.8092, 0.7931, 0.7786]
f1 = [0.9749, 0.9949, 0.8035, 0.7843, 0.7648]

results = pd.DataFrame({
    "Model": models,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
})

results.to_csv(
    "results/final_model_comparison.csv",
    index=False
)

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

for metric, values in metrics.items():
    plt.figure(figsize=(10, 6))
    plt.bar(models, values)
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.title("Plant Disease Detection - " + metric + " Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(
        "results/graphs/final_" + metric.lower().replace(" ", "_") + "_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

print("\nFINAL MODEL COMPARISON")
print("=" * 70)
print(results.to_string(index=False))
print("\nFinal comparison CSV saved.")
print("Final comparison graphs saved.")