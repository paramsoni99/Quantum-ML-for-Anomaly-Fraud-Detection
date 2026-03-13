# ⚛️ Quantum-Quest: Next-Gen Fraud Detection using Variational Quantum Classifiers (VQC)

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PennyLane](https://img.shields.io/badge/PennyLane-Quantum_ML-purple?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-orange?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?style=for-the-badge&logo=pandas)

## 📌 Project Overview
As financial transactions become increasingly complex, traditional machine learning models struggle to catch highly sophisticated fraudulent anomalies, especially in massively imbalanced datasets (where fraud makes up < 1% of the data). 

This project explores **Quantum Machine Learning (QML)** as a solution. By compressing transaction data and mapping it onto the Bloch sphere using **Angle Embedding**, we trained a 6-qubit **Variational Quantum Classifier (VQC)**. The goal? To prove that evaluating multidimensional data in quantum space can outperform classical industry-standard models like Logistic Regression.

> **Spoiler Alert:** The Quantum model destroyed the classical baseline by achieving a **+6.7% increase in F1-Score**.

---

## 📊 The Showdown: Classical vs. Quantum

We evaluated both models on a highly imbalanced dataset of 1,000,000 transactions. To ensure a fair fight, both models were trained on the exact same balanced subset using mathematically compressed Principal Component Analysis (PCA) features.

| Model | Architecture | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Classical Baseline** | Logistic Regression | 71.53% | ~0.72 | ~0.69 | **0.7090** |
| **Quantum AI** | 6-Qubit VQC (PennyLane) | 76.00% | 0.7261 | 0.8350 | **0.7767** |

**🏆 Conclusion:** The Quantum VQC demonstrated superior capability in isolating fraudulent anomalies, prioritizing higher recall and a significantly stronger F1-Score.

---

## 🏗️ Project Architecture & Notebooks

This repository is structured into three sequential Jupyter Notebooks, designed to be run in Google Colab for optimized compute.

### `01_data_exploration.ipynb` : Data Engineering & PCA
* Conducted rigorous Exploratory Data Analysis (EDA) on 1 million rows.
* Implemented **Isolation Forest** to analyze statistical outliers without destroying ground-truth data.
* Executed One-Hot Encoding and Standard Scaling.
* Leveraged **Principal Component Analysis (PCA)** to compress 30+ features down to 6 core dimensions, creating a "quantum-ready" state.

### `02_qml.ipynb` : The Classical Baseline
* Built an industry-standard Logistic Regression model to set the "score to beat."
* Utilized strict under-sampling to mitigate the 99-to-1 class imbalance.
* Generated baseline Confusion Matrices and ROC-AUC curves.

### `03_qml_v3.ipynb` : The Quantum Brain (VQC)
* Utilized **PennyLane** to construct a customized 6-qubit quantum simulator.
* Transformed the classical PCA features into quantum rotational angles ($0$ to $\pi$).
* Designed a quantum circuit using `StronglyEntanglingLayers`.
* Trained the model using a custom **Adam Optimizer** and Binary Cross Entropy (BCE) loss over 20 epochs.
* Calculated Youden's J statistic from the ROC curve to dynamically find the mathematically optimal classification threshold.

---

## 🛠️ Tech Stack & Libraries
* **Quantum Computation:** `PennyLane`, `PennyLane-Qiskit`
* **Machine Learning:** `Scikit-Learn` (PCA, Logistic Regression, Scalers, Metrics)
* **Data Manipulation:** `Pandas`, `NumPy`
* **Data Visualization:** `Matplotlib`, `Seaborn`
* **Environment:** Google Colab (Highly recommended for simulating quantum entanglement without local RAM overload).

---

## 🚀 How to Run the Project

1. Clone this repository to your local machine or Google Drive.
   ```bash
   git clone [https://github.com/your-username/quantum-quest.git](https://github.com/your-username/quantum-quest.git)