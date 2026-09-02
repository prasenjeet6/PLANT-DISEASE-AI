import streamlit as st
from PIL import Image

from src.predict import predict_disease
from reliability import check_reliability
from disease_info import disease_info
from src.rag_llm import generate_disease_explanation

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection")

st.write(
    "Upload a plant leaf image to identify the disease using a PSO optimized hybrid model."
)

st.sidebar.title("Project Details")
st.sidebar.write("Dataset: PlantVillage")
st.sidebar.write("Number of Classes: 38")
st.sidebar.write("Models: MobileNetV2 + EfficientNetV2B0")
st.sidebar.write("Optimization: Particle Swarm Optimization")
st.sidebar.write("MobileNetV2 Weight: 0.3745")
st.sidebar.write("EfficientNetV2B0 Weight: 0.6255")
st.sidebar.write("PSO Hybrid Accuracy: 98.67%")
st.sidebar.write("PSO Hybrid F1 Score: 98.66%")
st.sidebar.write("RAG + LLM: Ollama Llama 3.2")

st.divider()

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        width=450
    )

    with st.spinner("Analyzing the leaf..."):
        result = predict_disease(image)
        reliability = check_reliability(result)

    mobile_prediction = result["mobilenet_prediction"]
    mobile_confidence = result["mobilenet_confidence"]

    efficient_prediction = result["efficientnet_prediction"]
    efficient_confidence = result["efficientnet_confidence"]

    hybrid_prediction = result["hybrid_prediction"]
    hybrid_confidence = result["hybrid_confidence"]

    mobile_name = mobile_prediction.replace(
        "___", " - "
    ).replace("_", " ")

    efficient_name = efficient_prediction.replace(
        "___", " - "
    ).replace("_", " ")

    hybrid_name = hybrid_prediction.replace(
        "___", " - "
    ).replace("_", " ")

    st.divider()

    st.subheader("Final Hybrid Prediction")

    st.success(
        f"Detected Disease: {hybrid_name}"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Hybrid Confidence",
            f"{hybrid_confidence:.2f}%"
        )

    with col2:
        st.metric(
            "Model",
            "PSO Optimized Hybrid"
        )

    st.progress(
        min(hybrid_confidence / 100, 1.0)
    )

    st.write("MobileNetV2 Weight: 0.3745")
    st.write("EfficientNetV2B0 Weight: 0.6255")

    st.divider()

    st.subheader("Individual Model Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### MobileNetV2")
        st.write(mobile_name)
        st.write(
            f"Confidence: {mobile_confidence:.2f}%"
        )
        st.progress(
            min(mobile_confidence / 100, 1.0)
        )

    with col2:
        st.markdown("### EfficientNetV2B0")
        st.write(efficient_name)
        st.write(
            f"Confidence: {efficient_confidence:.2f}%"
        )
        st.progress(
            min(efficient_confidence / 100, 1.0)
        )

    st.divider()

    st.subheader("Prediction Reliability")

    if reliability["agreement"]:
        st.success(
            "Both individual models gave the same prediction."
        )
    else:
        st.warning(
            "The individual models gave different predictions."
        )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Reliability",
            reliability["reliability"]
        )

    with col2:
        st.metric(
            "Average Confidence",
            f"{reliability['average_confidence']:.2f}%"
        )

    st.divider()

    st.subheader("Grad-CAM Visualization")

    st.write(
        "The highlighted regions show the areas that influenced "
        "the individual model predictions."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### MobileNetV2")
        st.image(
            result["mobilenet_heatmap"],
            caption="MobileNetV2 Grad-CAM",
            use_container_width=True
        )

    with col2:
        st.markdown("### EfficientNetV2B0")
        st.image(
            result["efficientnet_heatmap"],
            caption="EfficientNetV2B0 Grad-CAM",
            use_container_width=True
        )

    info = disease_info.get(
        hybrid_prediction
    )

    if info:

        st.divider()

        st.subheader("Disease Information")

        st.write(
            info["description"]
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Treatment")
            st.write(
                info["treatment"]
            )

        with col2:
            st.markdown("### Prevention")
            st.write(
                info["prevention"]
            )

    else:
        st.info(
            "Disease information is not available."
        )

    st.divider()

    st.subheader("🤖 AI Disease Explanation")

    st.write(
        "The detected disease information is provided to the "
        "RAG system, which uses retrieved disease knowledge to "
        "generate an explanation using Ollama Llama 3.2."
    )

    with st.spinner("Generating AI explanation..."):
        try:
            ai_explanation = generate_disease_explanation(
                hybrid_prediction
            )

            if ai_explanation:
                st.info(ai_explanation)
            else:
                st.warning(
                    "The AI explanation could not be generated."
                )

        except Exception as e:
            st.error(
                f"AI explanation error: {str(e)}"
            )

    st.divider()

    st.subheader("Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "MobileNetV2",
            "88.84%"
        )

    with col2:
        st.metric(
            "EfficientNetV2B0",
            "98.46%"
        )

    with col3:
        st.metric(
            "50:50 Hybrid",
            "98.29%"
        )

    with col4:
        st.metric(
            "PSO Hybrid",
            "98.67%"
        )

    st.write(
        "The PSO optimized hybrid model uses a weight of 0.3745 "
        "for MobileNetV2 and 0.6255 for EfficientNetV2B0."
    )

else:
    st.info(
        "Upload a clear plant leaf image to start the prediction."
    )

st.divider()

st.subheader("About the Project")

st.write(
    "This project uses MobileNetV2 and EfficientNetV2B0 to detect "
    "plant diseases from leaf images. Their predictions are combined "
    "using optimized weights obtained through Particle Swarm Optimization. "
    "The system also provides confidence, reliability, Grad-CAM "
    "visualization, disease treatment and prevention information, "
    "and RAG-based AI-generated explanations using Ollama Llama 3.2."
)

st.caption(
    "TensorFlow • Keras • Streamlit • PlantVillage Dataset • PSO • RAG • Ollama Llama 3.2"
)