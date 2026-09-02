import os
import matplotlib.pyplot as plt

iterations = [1, 2, 3, 4, 5, 6]
best_accuracy = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
best_weight = [0.4140, 0.4140, 0.4140, 0.4140, 0.4140, 0.4140]

os.makedirs("results/graphs", exist_ok=True)

plt.figure(figsize=(8, 5))

plt.plot(
    iterations,
    best_accuracy,
    marker="o",
    label="Best Accuracy"
)

plt.xlabel("PSO Iteration")
plt.ylabel("Best Accuracy")
plt.title("PSO Optimization Convergence")
plt.xticks(iterations)
plt.ylim(0.99, 1.001)
plt.legend()
plt.grid(True)

plt.savefig(
    "results/graphs/pso_convergence.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.figure(figsize=(8, 5))

plt.plot(
    iterations,
    best_weight,
    marker="o",
    label="Best MobileNetV2 Weight"
)

plt.xlabel("PSO Iteration")
plt.ylabel("MobileNetV2 Weight")
plt.title("PSO Weight Optimization")
plt.xticks(iterations)
plt.ylim(0.40, 0.43)
plt.legend()
plt.grid(True)

plt.savefig(
    "results/graphs/pso_weight_optimization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("PSO convergence graph saved successfully.")
print("PSO weight graph saved successfully.")