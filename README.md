# CNN Digit Recognition

An interactive handwritten digit recognition system powered by a Convolutional Neural Network (CNN) trained on the MNIST dataset. The project allows users to draw digits on an interactive canvas and view real-time class predictions along with prediction confidence.

## Live Demo

- **Link:** [Try the live app](https://digitrecognition-cnn.streamlit.app/)

## Overview

This project implements an end-to-end deep learning pipeline for handwritten digit classification:
- **Model Architecture:** Convolutional Neural Network (CNN) utilizing convolutional layers, pooling, and dense layers for feature extraction and classification.
- **Dataset:** MNIST dataset of handwritten digits (0–9).
- **Interactive Interface:** A Streamlit web application with a live drawing canvas enabling users to sketch digits and receive instant inference.

## Repository Structure

- `app.py` - Interactive Streamlit web app with drawing canvas for real-time predictions.
- `tut.ipynb` - Jupyter notebook walking through data exploration, CNN architecture, training, evaluation, and model serialization.
- `cnn_model.joblib` - Ready-to-use pre-trained CNN model.

## Pre-trained Model & Quick Usage

To get started immediately without having to retrain the CNN model from scratch, the trained model has been included directly in this repository as `cnn_model.joblib`.

### Direct Download & Run

You can download/clone the repo and launch the web app with the pre-trained model right away:

```bash
# 1. Clone repository
git clone https://github.com/SlightlyBetterYuvraj/cnn-digit-recognition.git
cd cnn-digit-recognition

# 2. Install dependencies
pip install streamlit streamlit-drawable-canvas tensorflow keras joblib numpy pillow

# 3. Run the interactive app
streamlit run app.py
```

### Direct Python Usage Snippet

If you want to use the pre-trained model directly in your own Python scripts:

```python
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load("cnn_model.joblib")

# Input format: (batch_size, 28, 28, 1), normalized to [0, 1]
# Example: dummy input for inference
sample_input = np.zeros((1, 28, 28, 1), dtype=np.float32)

# Predict digit
probabilities = model.predict(sample_input)
predicted_digit = int(np.argmax(probabilities))
confidence = float(np.max(probabilities)) * 100

print(f"Predicted Digit: {predicted_digit} ({confidence:.2f}% confidence)")
```

## Development & Attribution

- **Machine Learning & Modeling:** The CNN architecture design, data preprocessing, training pipeline, and evaluation were developed independently.
- **UI Development:** AI was used exclusively to assist with UI development and frontend layout styling.

## Planned Features

- Live web deployment
- Real-time confidence score breakdown
- Support for customizable stroke thickness and canvas clearing
