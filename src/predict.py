import numpy as np
from tensorflow.keras.models import load_model

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess

from src.gradcam_plus_plus import generate_gradcam_pp, overlay_heatmap

mobilenet_model = load_model("models/mobilenet.keras")
efficientnet_model = load_model("models/efficientnet.keras")

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

MOBILE_WEIGHT = 0.3745
EFFICIENT_WEIGHT = 0.6255

def predict_disease(image):

    original_image = image.convert("RGB")
    resized_image = original_image.resize((224, 224))

    img = np.array(resized_image).astype("float32")

    mobile_input = mobilenet_preprocess(img.copy())
    mobile_input = np.expand_dims(mobile_input, axis=0)

    efficient_input = efficientnet_preprocess(img.copy())
    efficient_input = np.expand_dims(efficient_input, axis=0)

    mobile_prediction = mobilenet_model.predict(
        mobile_input,
        verbose=0
    )[0]

    efficient_prediction = efficientnet_model.predict(
        efficient_input,
        verbose=0
    )[0]

    hybrid_prediction = (
        MOBILE_WEIGHT * mobile_prediction
        + EFFICIENT_WEIGHT * efficient_prediction
    )

    mobile_index = int(np.argmax(mobile_prediction))
    efficient_index = int(np.argmax(efficient_prediction))
    hybrid_index = int(np.argmax(hybrid_prediction))

    mobile_confidence = float(
        np.max(mobile_prediction) * 100
    )

    efficient_confidence = float(
        np.max(efficient_prediction) * 100
    )

    hybrid_confidence = float(
        np.max(hybrid_prediction) * 100
    )

    mobile_heatmap = generate_gradcam_pp(
        mobilenet_model,
        mobile_input,
        mobile_index,
        "out_relu"
    )

    efficient_heatmap = generate_gradcam_pp(
        efficientnet_model,
        efficient_input,
        efficient_index,
        "top_activation"
    )

    mobile_overlay = overlay_heatmap(
        original_image,
        mobile_heatmap
    )

    efficient_overlay = overlay_heatmap(
        original_image,
        efficient_heatmap
    )

    return {
        "mobilenet_prediction": class_names[mobile_index],
        "mobilenet_confidence": mobile_confidence,
        "efficientnet_prediction": class_names[efficient_index],
        "efficientnet_confidence": efficient_confidence,
        "hybrid_prediction": class_names[hybrid_index],
        "hybrid_confidence": hybrid_confidence,
        "mobile_weight": MOBILE_WEIGHT,
        "efficient_weight": EFFICIENT_WEIGHT,
        "mobilenet_heatmap": mobile_overlay,
        "efficientnet_heatmap": efficient_overlay
    }