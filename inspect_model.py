from tensorflow.keras.models import load_model

model = load_model("models/mobilenet.keras", compile=False)

print("Model:", model.name)
print()

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.output.shape)