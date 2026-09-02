import numpy as np
import tensorflow as tf
import cv2


def generate_gradcam_pp(model, image_tensor, class_index, layer_name):
    """
    Generate a Grad-CAM++ heatmap for a Keras model.
    """

    # Get target convolutional layer
    target_layer = model.get_layer(layer_name)

    # Create a model that returns:
    # 1. feature maps from target layer
    # 2. final model predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            target_layer.output,
            model.output
        ]
    )

    image_tensor = tf.cast(image_tensor, tf.float32)

    with tf.GradientTape(persistent=True) as tape3:
        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape(persistent=True) as tape1:

                conv_outputs, predictions = grad_model(image_tensor)

                # Target class score
                class_score = predictions[:, class_index]

            first_grad = tape1.gradient(
                class_score,
                conv_outputs
            )

        second_grad = tape2.gradient(
            first_grad,
            conv_outputs
        )

    third_grad = tape3.gradient(
        second_grad,
        conv_outputs
    )

    # Clean up tapes
    del tape1
    del tape2
    del tape3

    if first_grad is None:
        raise RuntimeError(
            "Grad-CAM failed: first gradient is None."
        )

    # If higher derivatives are unavailable,
    # fall back safely.
    if second_grad is None:
        second_grad = tf.zeros_like(first_grad)

    if third_grad is None:
        third_grad = tf.zeros_like(first_grad)

    conv_outputs = conv_outputs[0]
    first_grad = first_grad[0]
    second_grad = second_grad[0]
    third_grad = third_grad[0]

    # Grad-CAM++ alpha calculation
    numerator = second_grad

    denominator = (
        2.0 * second_grad
        + third_grad * conv_outputs
    )

    denominator = tf.where(
        tf.abs(denominator) < 1e-7,
        tf.ones_like(denominator),
        denominator
    )

    alpha = numerator / denominator

    # Positive gradients only
    positive_gradients = tf.maximum(first_grad, 0.0)

    weights = tf.reduce_sum(
        alpha * positive_gradients,
        axis=(0, 1)
    )

    # Weighted combination of feature maps
    heatmap = tf.reduce_sum(
        conv_outputs * weights,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(heatmap, 0)

    # Normalize
    max_value = tf.reduce_max(heatmap)

    if float(max_value) > 0:
        heatmap = heatmap / max_value

    heatmap = heatmap.numpy()

    return heatmap


def overlay_heatmap(image, heatmap, alpha=0.35):
    """
    Overlay Grad-CAM heatmap on original image.
    """

    # Convert PIL image to numpy
    image_np = np.array(image).copy()

    # Ensure RGB
    if image_np.ndim == 2:
        image_np = cv2.cvtColor(
            image_np,
            cv2.COLOR_GRAY2RGB
        )

    # Resize heatmap
    heatmap = cv2.resize(
        heatmap,
        (image_np.shape[1], image_np.shape[0])
    )

    # Normalize safely
    heatmap = np.maximum(heatmap, 0)

    max_value = np.max(heatmap)

    if max_value > 0:
        heatmap = heatmap / max_value

    # Convert to 8-bit
    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    # Apply color map
    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # OpenCV uses BGR, convert to RGB
    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    # Blend
    overlay = (
        (1 - alpha) * image_np
        + alpha * colored_heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    return overlay