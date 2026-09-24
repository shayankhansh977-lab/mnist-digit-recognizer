# 🔢 Handwritten Digit Recognition App

An end-to-end Machine Learning web application that predicts handwritten digits (0–9) using a Neural Network model trained on the MNIST dataset.
The interactive frontend is built using **Streamlit** with real-time image preprocessing controls.

---

## 🚀 Features

* **Pre-trained Neural Network**: Built with TensorFlow/Keras and saved as `mnist_model.h5`.
* **Interactive Web UI**: Streamlit interface with image upload support (PNG, JPG, JPEG).
* **Real-time Image Adjustment**: Custom sliders for brightness, contrast, and background thresholding.
* **Auto-Inversion Pipeline**: Automatically handles white-background and dark-background images.
* **Fun Celebrations**: Interactive balloons effect upon successful digit prediction.

---

## 🛠️ Tech Stack

* **Programming Language**: Python
* **Machine Learning Framework**: TensorFlow / Keras
* **Data Processing**: NumPy, Pillow (PIL)
* **Web Framework**: Streamlit

---

## 📂 Project Structure

```text
mnist-project/
│
├── app.py              # Streamlit Web Application
├── main.py             # Model training and initial setup script
├── mnist_model.h5      # Trained Keras model file
└── requirements.txt    # Required dependencies for deployment
