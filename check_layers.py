from tensorflow.keras.models import load_model

model = load_model("models/efficientnet.keras")

for i, layer in enumerate(model.layers):
    print(i, layer.name)