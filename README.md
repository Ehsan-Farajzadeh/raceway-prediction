# raceway-prediction
Deep learning prediction of raceway size in a blast furnace using CFD thermal images
# Forecasting Raceway Dimensions in a Blast Furnace Using Deep Learning

This project presents a hybrid deep learning approach to predict the **temporal evolution of raceway dimensions** (depth and height) inside an **ironmaking blast furnace** using **thermal images** generated from **CFD simulations**. By combining the spatial feature extraction capability of **Convolutional Neural Networks (CNN)** with the temporal sequence modeling power of **Long Short-Term Memory (LSTM)** networks, we built accurate models for **real-time monitoring** of blast furnace behavior.

---

## 🔍 Motivation

In blast furnaces, the raceway plays a critical role in combustion and heat transfer. Monitoring its size helps optimize:
- Fuel injection strategies
- Energy efficiency
- Furnace lifespan

However, **direct measurements** are impractical due to the extreme operating environment. CFD simulations offer a detailed alternative, but they are computationally expensive. This project uses deep learning to learn from CFD-derived thermal images and enable **fast, accurate, and scalable prediction** of raceway size in real time.

---

## 🧠 Methods

We developed and compared two deep learning models:

- **LSTM**: Processes sequential thermal image data to predict raceway depth and height over time.
- **CNN-LSTM**: First extracts spatial features with 1D convolutions and then models temporal dependencies with LSTM layers.

⚠️ Note: The dataset used in this project (LSTM.csv) is derived from CFD simulations and contains proprietary results, and therefore is not publicly shared. 

To test the code:
- You may use the `preprocessing.py` script with your own thermal images.
- Or modify the models to work with synthetic/sample data.

---

## 📁 Project Structure
