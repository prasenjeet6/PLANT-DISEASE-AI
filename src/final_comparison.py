import os
import matplotlib.pyplot as plt

models = [
    "MobileNetV2",
    "EfficientNetV2B0",
    "Hybrid\n50:50",
    "Optimized Hybrid\n40:60"
]

accuracy = [0.9751, 0.9949, 0.9950, 0.9957]
precision = [0.9751, 0.9949, 0.9950, 0.9957]
recall = [0.9751, 0.9949, 0.9950, 0.9957]
f1 = [0.9751, 0.9949, 0.9950, 0.9957]

os.makedirs("results/final", exist_ok=True)

with open("results/final/model_comparison.txt", "w") as f:
    f.write("MODEL PERFORMANCE COMPARISON\n")
    f.write("============================\n\n")

    for i, model in enumerate(models):
        f.write(f"{model.replace(chr(10), ' ')}\n")
        f.write(f"Accuracy : {accuracy[i]:.4f}\n")
        f.write(f"Precision: {precision[i]:.4f}\n")
        f.write(f"Recall   : {recall[i]:.4f}\n")
        f.write(f"F1 Score : {f1[i]:.4f}\n\n")

fig, ax = plt.subplots(figsize=(12, 7))

x = range(len(models))
width = 0.2

ax.bar([i - 1.5 * width for i in x], accuracy, width, label="Accuracy")
ax.bar([i - 0.5 * width for i in x], precision, width, label="Precision")
ax.bar([i + 0.5 * width for i in x], recall, width, label="Recall")
ax.bar([i + 1.5 * width for i in x], f1, width, label="F1 Score")

ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_title("Performance Comparison of Plant Disease Classification Models")
ax.set_xticks(list(x))
ax.set_xticklabels(models)
ax.set_ylim(0.94, 1.01)
ax.legend()

plt.tight_layout()

plt.savefig(
    "results/final/model_comparison.png",
    dpi=300
)

plt.show()

print("\n===================================")
print("Final Model Comparison Completed")
print("===================================")
print("Best Model: Optimized Hybrid")
print("MobileNetV2 Weight: 0.4")
print("EfficientNetV2B0 Weight: 0.6")
print("Accuracy: 0.9957")
print("Results saved in results/final/")