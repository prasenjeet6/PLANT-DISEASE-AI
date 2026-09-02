# AI-Based Plant Disease Detection

## Description

This project detects plant leaf diseases using a MobileNetV2 deep learning model trained on the PlantVillage dataset.

Users can upload a plant leaf image, and the application predicts the disease along with confidence, description, treatment, and prevention measures.

## Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- MobileNetV2
- PlantVillage Dataset

## Dataset

PlantVillage Dataset (Kaggle)

## Features

- Plant Disease Detection
- Confidence Score
- Disease Description
- Treatment Suggestions
- Prevention Measures

## Run Project

```bash
streamlit run app.py
```

## Dataset

The complete Plant Disease Dataset is provided separately because the dataset is approximately 8 GB and is not stored directly in this GitHub repository.

### Download Dataset

[Download PlantVillage_Color.zip from Google Drive](https://drive.google.com/file/d/1Vu5WeeqoCH_ACFfSSd0bTpe1YT0Ktxgb/view?usp=sharing)

### Dataset Setup

After downloading the dataset, place/extract it into the dataset folder of this project.

Project structure:

    PLANT-DISEASE-AI/
    +-- dataset/
    +-- models/
    +-- notebooks/
    +-- results/
    +-- src/
