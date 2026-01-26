# Acoustic Scene Classification using Deep Learning

## Introduction
This project focuses on **Acoustic Scene Classification (ASC)** using deep learning models such as **ANN, CNN, and CRNN**.  
ASC aims to classify environmental audio recordings into predefined acoustic scenes, enabling machines to understand and interpret real-world audio contexts.

The system is trained and evaluated on the **UrbanSound8K dataset**, containing urban environmental sounds such as sirens, gunshots, drilling, and street music.  
The project also includes **real-time inference** and a **GUI-based audio upload interface**.

---

## Dataset
### UrbanSound8K
- 8,732 labeled audio clips (≤ 4 seconds)
- 10 classes:
  - Air Conditioner
  - Car Horn
  - Children Playing
  - Dog Bark
  - Drilling
  - Engine Idling
  - Gun Shot
  - Jackhammer
  - Siren
  - Street Music
- Sampling rate: 44.1 kHz
- Dataset split: **80% training / 20% testing**

Dataset link:  
https://urbansounddataset.weebly.com/download-urbansound8k.html

---

## Feature Extraction
- Audio loading and resampling using **Librosa**
- Extraction of **40 MFCC coefficients**
- Temporal averaging of MFCC features
- Label encoding and one-hot encoding
- Input reshaped to match deep learning model requirements

---

## Models Implemented

### 1. Artificial Neural Network (ANN)
- Input: 40-dimensional MFCC feature vector
- Dense layers: 100 → 200 → 100 neurons
- Activation: ReLU
- Dropout: 0.5
- Output: Softmax (10 classes)

### 2. Convolutional Neural Network (CNN)
- Two Conv2D layers (64, 128 filters)
- Kernel size: (3 × 3)
- MaxPooling and Dropout for regularization
- Flatten + Dense layers
- Softmax output layer

### 3. Convolutional Recurrent Neural Network (CRNN)
- Conv2D layers for spatial feature extraction
- GRU/LSTM layer for temporal modeling
- Captures both **spectral and temporal dependencies**
- Best-performing model in this project

---

## Algorithm Pipeline
1. Load audio file and resample
2. Extract MFCC features
3. Normalize and reshape features
4. Train ANN / CNN / CRNN models
5. Evaluate performance using accuracy and confusion matrix
6. Deploy trained CRNN for:
   - GUI-based prediction
   - Real-time acoustic scene classification

---

## Results

### Model Performance (UrbanSound8K)

| Model | Accuracy | Remarks |
|------|----------|---------|
| ANN  | Moderate | Suitable for baseline comparison |
| CNN  | High     | Strong spatial feature learning |
| CRNN | Highest  | Best temporal + spectral modeling |

CRNN demonstrated superior performance due to its ability to model temporal dependencies in audio signals.

---

## GUI & Real-Time Demo
- GUI allows users to upload an audio file and view predicted scene
- Real-time ASC implemented using microphone input
- Pre-trained CRNN model saved in `.h5` format

---

## Dependencies
- Python 3.x
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
