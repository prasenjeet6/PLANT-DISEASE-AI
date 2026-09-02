import tensorflow as tf
import numpy as np
import cv2

def generate_gradcam(model, img_array, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.nn.relu(heatmap)

    max_val = tf.reduce_max(heatmap)

    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_heatmap(image, heatmap):

    image = np.array(image)

    heatmap = cv2.resize(
        heatmap,
        (image.shape[1], image.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.GaussianBlur(
        heatmap,
        (11,11),
        0
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_TURBO
    )

    overlay = cv2.addWeighted(
        image,
        0.65,
        heatmap,
        0.35,
        0
    )

    return overlay