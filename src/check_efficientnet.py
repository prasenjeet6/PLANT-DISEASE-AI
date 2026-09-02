from tensorflow.keras.models import load_model

model = load_model("models/efficientnet.keras")

print("Optimizer:", model.optimizer)
print("Loss:", model.loss)
print("Metrics:", model.metrics_names)