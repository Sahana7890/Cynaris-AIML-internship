# Model Card — Chest X-Ray Classification

## Model

Planned model: ResNet-18 using transfer learning.

## Intended Use

The model is intended for educational and research purposes to investigate automated classification of chest X-ray images into Normal, Pneumonia, and COVID-19 categories.

## Classes

1. Normal
2. Pneumonia
3. COVID-19

## Planned Evaluation

The model will be evaluated using:

- ROC-AUC
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- COVID-19 false-negative rate

## Limitations

The model should not be considered a clinical diagnostic system.

Performance may be affected by:

- Dataset size and quality
- Class imbalance
- Differences between training and real-world X-ray images
- Image acquisition equipment
- Patient population differences
- Dataset bias
- Incorrect or noisy labels

Grad-CAM visualizations provide model explanations but should not be interpreted as definitive medical evidence.

## Responsible Use

The system should only be used as an educational/research prototype. Any real clinical deployment would require extensive validation, appropriate regulatory review, clinical oversight, and testing on representative external datasets.
