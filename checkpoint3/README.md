# Checkpoint 3 — Core Model + Experimentation

## Overview

Checkpoint 3 focuses on training the primary chest X-ray classification model
and comparing different hyperparameter configurations.

The primary model is a **ResNet-18 transfer-learning model** with three
classification classes:

- Normal
- Pneumonia
- COVID-19

At least two different hyperparameter configurations are trained and tracked
using **MLflow**.

---

## Checkpoint 3 Requirements

The objective is to:

1. Train the primary ResNet-18 model.
2. Train the model using at least two hyperparameter configurations.
3. Track experiments using MLflow.
4. Compare the experimental results.
5. Select the best-performing configuration.
6. Provide evidence supporting the selected configuration.

---

## Project Structure

```text
checkpoint3/
│
├── README.md
├── experiment_config.py
├── mlflow_train.py
├── compare_experiments.py
└── requirements.txt

The Checkpoint 3 files work together with the Checkpoint 2 files:

src/
├── config.py
├── preprocessing.py
├── dataset.py
├── model.py
├── train.py
└── evaluate.py
