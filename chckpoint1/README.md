# Chest X-Ray Classification Using ResNet-18

## Project Overview

This project aims to develop a deep learning system for classifying chest X-ray images into three classes:

- Normal
- Pneumonia
- COVID-19

The project uses transfer learning with a pretrained ResNet-18 convolutional neural network.

## Objective

The main objectives are:

1. Train a three-class chest X-ray classification model.
2. Use ResNet-18 transfer learning.
3. Evaluate the model using appropriate classification metrics.
4. Generate Grad-CAM heatmaps to visualize important image regions.
5. Build a local Gradio application for image prediction.
6. Document model limitations and responsible-use considerations.

## Planned Architecture

```text
Chest X-Ray Image
        |
        v
Image Preprocessing
        |
        v
ResNet-18 Transfer Learning
        |
        v
Three-Class Classifier
        |
        +--------------------+
        |                    |
        v                    v
Class Probabilities      Grad-CAM
        |                    |
        +---------+----------+
                  |
                  v
             Gradio Demo
```

## Classes

The model will classify images into:

- Normal
- Pneumonia
- COVID-19

## Evaluation Metrics

The planned evaluation metrics are:

- ROC-AUC
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- COVID-19 False-Negative Rate

Target requirements:

- Test AUC >= 0.87
- COVID-19 false-negative rate < 15%
- Grad-CAM heatmaps should render for all three classes.

## Technology Stack

- Python
- PyTorch
- ResNet-18
- scikit-learn
- Grad-CAM
- Gradio
- NumPy
- Pandas
- Matplotlib

## Project Structure

```text
chest-xray-classification/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── data_pipeline.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
│
├── models/
├── results/
├── app.py
├── requirements.txt
├── MODEL_CARD.md
└── README.md
```

## Planned Workflow

1. Prepare and inspect the dataset.
2. Clean and preprocess the images.
3. Split the data into training, validation, and test sets.
4. Load pretrained ResNet-18.
5. Replace the final classification layer with a three-class classifier.
6. Train and validate the model.
7. Evaluate the model on the test set.
8. Generate Grad-CAM explanations.
9. Build the Gradio interface.
10. Document limitations in the model card.

## Disclaimer

This project is an educational/research prototype. It is not intended to provide medical diagnosis or replace qualified medical professionals.
