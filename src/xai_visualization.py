import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess

IMG_SIZE = (224, 224)

DATASET_PATH = "dataset/color"
MOBILENET_PATH = "models/mobilenet.keras"
EFFICIENTNET_PATH = "models/efficientnet.keras"

OUTPUT_PATH = "results/graphs/xai_gradcam_plus_plus_5_samples.png"

samples = [
    "Apple___Apple_scab",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Potato___Late_blight",
    "Tomato___Early_blight"
]

mobilenet_model = load_model(MOBILENET_PATH)
efficientnet_model = load_model(EFFICIENTNET_PATH)

def find_target_layer(model):
    for layer in reversed(model.layers):
        try:
            output_shape = layer.output.shape
            if len(output_shape) == 4:
                return layer.name
        except:
            continue
    raise ValueError("No suitable convolutional layer found.")

def generate_gradcam_pp(model, image_tensor, class_index, layer_name):
    target_layer = model.get_layer(layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    image_tensor = tf.cast(image_tensor, tf.float32)

    with tf.GradientTape(persistent=True) as tape3:
        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape(persistent=True) as tape1:
                conv_outputs, predictions = grad_model(image_tensor)
                class_score = predictions[:, class_index]

            first_grad = tape1.gradient(class_score, conv_outputs)

        second_grad = tape2.gradient(first_grad, conv_outputs)

    third_grad = tape3.gradient(second_grad, conv_outputs)

    del tape1
    del tape2
    del tape3

    if first_grad is None:
        raise RuntimeError("Grad-CAM++ first gradient is None.")

    if second_grad is None:
        second_grad = tf.zeros_like(first_grad)

    if third_grad is None:
        third_grad = tf.zeros_like(first_grad)

    conv_outputs = conv_outputs[0]
    first_grad = first_grad[0]
    second_grad = second_grad[0]
    third_grad = third_grad[0]

    denominator = (
        2.0 * second_grad
        + third_grad * conv_outputs
    )

    denominator = tf.where(
        tf.abs(denominator) < 1e-7,
        tf.ones_like(denominator),
        denominator
    )

    alpha = second_grad / denominator

    positive_gradients = tf.maximum(first_grad, 0.0)

    weights = tf.reduce_sum(
        alpha * positive_gradients,
        axis=(0, 1)
    )

    heatmap = tf.reduce_sum(
        conv_outputs * weights,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)

    if float(max_value) > 0:
        heatmap = heatmap / max_value

    return heatmap.numpy()

def create_overlay(image, heatmap, alpha=0.4):
    image_np = np.array(image).astype(np.uint8)

    heatmap = cv2.resize(
        heatmap,
        (image_np.shape[1], image_np.shape[0])
    )

    heatmap = np.maximum(heatmap, 0)

    max_value = np.max(heatmap)

    if max_value > 0:
        heatmap = heatmap / max_value

    heatmap_uint8 = np.uint8(255 * heatmap)

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = (
        (1 - alpha) * image_np
        + alpha * colored_heatmap
    )

    return np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

def load_sample(class_name):
    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    files = []

    for file in os.listdir(class_path):
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            files.append(file)

    files.sort()

    if not files:
        raise FileNotFoundError(
            f"No image found for {class_name}"
        )

    return os.path.join(
        class_path,
        files[0]
    )

mobilenet_layer = find_target_layer(
    mobilenet_model
)

efficientnet_layer = find_target_layer(
    efficientnet_model
)

print("MobileNetV2 target layer:", mobilenet_layer)
print("EfficientNetV2B0 target layer:", efficientnet_layer)

os.makedirs(
    "results/graphs",
    exist_ok=True
)

fig, axes = plt.subplots(
    5,
    3,
    figsize=(15, 25)
)

for row, class_name in enumerate(samples):

    image_path = load_sample(class_name)

    original_image = Image.open(
        image_path
    ).convert("RGB")

    resized_image = original_image.resize(
        IMG_SIZE
    )

    image_array = np.array(
        resized_image
    ).astype(np.float32)

    mobile_input = mobilenet_preprocess(
        image_array.copy()
    )

    efficient_input = efficientnet_preprocess(
        image_array.copy()
    )

    mobile_input = np.expand_dims(
        mobile_input,
        axis=0
    )

    efficient_input = np.expand_dims(
        efficient_input,
        axis=0
    )

    mobile_prediction = mobilenet_model.predict(
        mobile_input,
        verbose=0
    )

    efficient_prediction = efficientnet_model.predict(
        efficient_input,
        verbose=0
    )

    mobile_class = int(
        np.argmax(mobile_prediction[0])
    )

    efficient_class = int(
        np.argmax(efficient_prediction[0])
    )

    mobile_heatmap = generate_gradcam_pp(
        mobilenet_model,
        mobile_input,
        mobile_class,
        mobilenet_layer
    )

    efficient_heatmap = generate_gradcam_pp(
        efficientnet_model,
        efficient_input,
        efficient_class,
        efficientnet_layer
    )

    mobile_overlay = create_overlay(
        resized_image,
        mobile_heatmap
    )

    efficient_overlay = create_overlay(
        resized_image,
        efficient_heatmap
    )

    axes[row, 0].imshow(
        resized_image
    )

    axes[row, 1].imshow(
        mobile_overlay
    )

    axes[row, 2].imshow(
        efficient_overlay
    )

    axes[row, 0].set_ylabel(
        class_name.replace("___", "\n"),
        fontsize=9
    )

    axes[row, 0].set_title(
        "Original Image",
        fontsize=11
    )

    axes[row, 1].set_title(
        "MobileNetV2 Grad-CAM++",
        fontsize=11
    )

    axes[row, 2].set_title(
        "EfficientNetV2B0 Grad-CAM++",
        fontsize=11
    )

    axes[row, 0].axis("off")
    axes[row, 1].axis("off")
    axes[row, 2].axis("off")

    print(
        f"{class_name} completed | "
        f"MobileNet class: {mobile_class} | "
        f"EfficientNet class: {efficient_class}"
    )

plt.suptitle(
    "XAI Visualization Using Grad-CAM++",
    fontsize=18,
    y=0.995
)

plt.tight_layout(
    rect=[0, 0, 1, 0.985]
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("XAI visualization completed successfully.")
print("5 representative images processed.")
print("Graph saved to:")
print(OUTPUT_PATH)