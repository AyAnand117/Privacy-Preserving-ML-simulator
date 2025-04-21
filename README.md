
# 🔒 Privacy-Preserving ML Simulator

A Streamlit-based interactive simulator for experimenting with privacy-preserving machine learning techniques, including **federated learning** and **differential privacy**. Designed for education, experimentation, and analysis of the privacy-utility tradeoff in ML workflows.

---

## 🚀 Features

### 🌍 Federated Learning Simulation
- Simulates data split across 3 clients.
- Each client trains a **local logistic regression** model independently.
- Models are aggregated using **federated averaging**.

### 🧂 Differential Privacy Integration
- Adds **Gaussian noise** to local model weights via an `ε` (epsilon) privacy parameter.
- **Interactive slider** to adjust privacy level and observe tradeoffs.

### 📊 Privacy-Utility Tradeoff Visualization
- Tracks **F1, Precision, and Recall** across different privacy levels.
- Live line chart shows model performance degradation as privacy increases.
- Uses `matplotlib` for detailed and responsive visualizations.

### ⚙️ Deep Learning with Opacus + PyTorch
- Train a custom **MLP model** using PyTorch and [Opacus](https://opacus.ai/).
- Tracks **epsilon progression per epoch**.
- Generates privacy budget reports and classification metrics.

### 📁 Streamlit-Powered UI
- Upload your own `creditcard.csv` or use a sample dataset.
- Interactive controls for:
  - **Epsilon (privacy level)**
  - **Noise multiplier (Opacus)**
  - **Epochs and batch size**
- Real-time metric updates and visual feedback.

### 🧠 Evaluation Metrics
- Supports:
  - **Precision**
  - **Recall**
  - **F1 Score**
- Includes `scikit-learn` classification reports for deep learning models.

### 🗃️ Modular Codebase
- `app.py`: UI logic via Streamlit
- `utils.py`: Helper functions for federated learning and DP
- `models.py`: MLP architecture
- `dp_training.py`: Opacus-based training pipeline

---

## 🛠️ Built With

- **Python**
- **Streamlit**
- **PyTorch**
- **Opacus**
- **scikit-learn**
- **matplotlib**
- **pandas / NumPy**

---


