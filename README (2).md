# 🧠 Interactive ML Pipeline Dashboard
### CS-303B · Machine Learning & ANN · CA-2 Project Exhibition

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn)](https://scikit-learn.org)

---

## 📌 Project Overview

A **production-ready, end-to-end Machine Learning web application** built with Python and Streamlit. The dashboard provides a complete AutoML pipeline — from raw CSV upload to trained model download — with a professional dark/light UI, interactive Plotly charts, and step-by-step guided workflow.

---

## ✨ Features

| Step | Feature |
|------|---------|
| 1 | **Data Input** — CSV upload, dataset preview, PCA 2D/3D visualisation |
| 2 | **EDA** — Missing values, distributions, correlation heatmap, target analysis |
| 3 | **Preprocessing** — Imputation, label/one-hot encoding, StandardScaler/MinMaxScaler |
| 4 | **Outlier Detection** — IQR, Isolation Forest, DBSCAN with interactive removal |
| 5 | **Feature Selection** — Variance Threshold, Correlation, Information Gain |
| 6 | **Data Split** — Adjustable train/test ratio with visual breakdown |
| 7 | **Model Selection** — Logistic/Linear Regression, SVM (kernel options), Random Forest, KMeans |
| 8 | **Training & CV** — K-Fold cross validation with fold-by-fold chart |
| 9 | **Metrics** — Classification (Acc, Prec, Recall, F1, CM) / Regression (MAE, MSE, RMSE, R²) + overfitting detection |
| 10 | **Hyperparameter Tuning** — GridSearchCV / RandomizedSearchCV with before/after comparison |
| ⭐ | **Model Download** — Download trained `.pkl` file |
| ⭐ | **Predict on New Data** — Upload CSV, get predictions, download results |
| ⭐ | **Dark / Light Theme** — Toggle in sidebar |

---

## 🗂 Project Structure

```
ml_dashboard/
│
├── pipeline.py          ← Main Streamlit application
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

---

## 🚀 Quick Start (Local)

### 1. Clone / Download

```bash
git clone https://github.com/<your-username>/ml-pipeline-dashboard.git
cd ml-pipeline-dashboard
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run pipeline.py
```

The app opens automatically at **http://localhost:8501**

---

## ☁️ Deploy to Streamlit Cloud (Free)

Follow the steps from the CA-2 tutorial PDF:

**Step 1 — Push to GitHub**
```bash
git init
git add .
git commit -m "ML Pipeline Dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

**Step 2 — Create Streamlit Account**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Sign up / log in with GitHub

**Step 3 — Deploy**
1. Click **"Create app"** (top right)
2. Choose **"Deploy a public app from GitHub"**
3. Fill in:
   - **Repository:** `<your-username>/<repo-name>`
   - **Branch:** `main`
   - **Main file path:** `pipeline.py`
4. Click **Deploy!**

Your app gets a public URL like: `https://<your-app>.streamlit.app`

---

## 📊 Supported Models

### Classification
| Model | Notes |
|-------|-------|
| Logistic Regression | Adjustable C parameter |
| SVM | rbf / linear / poly / sigmoid kernels |
| Random Forest | Adjustable trees and max depth |
| KMeans | Unsupervised clustering |

### Regression
| Model | Notes |
|-------|-------|
| Linear Regression | Ordinary least squares |
| SVM (SVR) | Kernel-based regression |
| Random Forest | Ensemble regression |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.32 | Web UI framework |
| pandas | ≥2.0 | Data manipulation |
| numpy | ≥1.24 | Numerical computing |
| scikit-learn | ≥1.4 | ML algorithms |
| matplotlib | ≥3.7 | Base plotting |
| seaborn | ≥0.12 | Statistical plots |
| plotly | ≥5.18 | Interactive charts |

---

## 🎓 Course Information

- **Course:** CS-303B — Machine Learning & Artificial Neural Networks
- **Assessment:** Continuous Assessment 2 (CA-2) — Project Exhibition
- **Marks:** 6 Marks
- **Coverage:** Unit 1 (KNN, Regression, SVM, Metrics) · Unit 2 (Feature Engineering, Outlier Detection, Clustering, PCA) · Unit 3 (Ensemble Learning, Hyperparameter Tuning, AutoML)

---

## 📋 How to Use the App

1. **Select problem type** (Classification / Regression) in the sidebar
2. **Upload a CSV** in the Data Input tab (or load a sample dataset)
3. **Select your target column** and click Confirm
4. Navigate through each tab in order — the stepper bar shows your progress
5. After training, download your `.pkl` model or upload new data for predictions

---

## 💡 Tips

- Use the **sample datasets** (Iris, California Housing) to test without uploading a file
- The **stepper bar** at the top tracks your progress through the pipeline
- All charts are **interactive** — hover, zoom, and pan with Plotly
- Toggle **Dark / Light mode** in the sidebar anytime

---

*Happy Coding! 🚀*
