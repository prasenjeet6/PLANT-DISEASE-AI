import os
import tensorflow as tf

MOBILENET_PATH = "models/mobilenet.keras"
EFFICIENTNET_PATH = "models/efficientnet.keras"

print("=" * 60)
print("MODEL CHECK")
print("=" * 60)

if not os.path.exists(MOBILENET_PATH):
    raise FileNotFoundError(f"Missing model: {MOBILENET_PATH}")

if not os.path.exists(EFFICIENTNET_PATH):
    raise FileNotFoundError(f"Missing model: {EFFICIENTNET_PATH}")

print("\nLoading MobileNetV2...")
mobilenet = tf.keras.models.load_model(MOBILENET_PATH)

print("MobileNetV2 loaded.")

print("\nLoading EfficientNetV2B0...")
efficientnet = tf.keras.models.load_model(EFFICIENTNET_PATH)

print("EfficientNetV2B0 loaded.")

print("\n" + "=" * 60)
print("MobileNetV2")
print("=" * 60)

print("Input shape:", mobilenet.input_shape)
print("Output shape:", mobilenet.output_shape)
print("Number of layers:", len(mobilenet.layers))

print("\nLast 15 MobileNetV2 layers:")

for layer in mobilenet.layers[-15:]:
    print(layer.name)

print("\n" + "=" * 60)
print("EfficientNetV2B0")
print("=" * 60)

print("Input shape:", efficientnet.input_shape)
print("Output shape:", efficientnet.output_shape)
print("Number of layers:", len(efficientnet.layers))

print("\nLast 15 EfficientNetV2B0 layers:")

for layer in efficientnet.layers[-15:]:
    print(layer.name)

print("\n" + "=" * 60)
print("MODEL CHECK COMPLETE")
print("=" * 60)