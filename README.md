# 🚀 ClarityML – Automated Machine Learning Web App

ClarityML is a simple yet powerful **AutoML web application built with Streamlit and PyCaret**.  
It allows users to upload datasets, perform automated data analysis, train machine learning models, evaluate performance, and export models/reports — all without writing code.

---

## ✨ Features

### ✔ AutoML Pipeline
Automatically handles the entire machine learning workflow including data preprocessing, model training, evaluation, and selection of the best performing model using PyCaret.

---

### ✔ Model Comparison
Compares multiple machine learning models and ranks them based on performance metrics like accuracy, precision, recall, and F1-score to help select the best model.

---

### ✔ Confusion Matrix + ROC Curve
Visualizes model performance using confusion matrix and ROC-AUC curves to evaluate classification accuracy and class-wise prediction quality.

#### 📊 Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)

#### 📈 ROC Curve
![ROC Curve](assets/roc_curve.png)

---

### ✔ Model Export (.pkl)
Allows users to download the trained machine learning model as a `.pkl` file for reuse in other applications or deployment environments.

---

### ✔ Report Generation
Generates a complete HTML-based machine learning report including model summary, performance metrics, and comparison results.

#### 📄 ML Report Preview
![ML Report](assets/ml_report.png)

---

### ✔ Data Profiling
Provides automated exploratory data analysis (EDA) using `ydata-profiling`, including data distribution, correlations, missing values, and insights with a single click.

---

## 🛠 Tech Stack

- Python  
- Streamlit ⚡  
- PyCaret   
- Pandas 📊  
- ydata-profiling 📈  
- Scikit-learn 🔬  

---

## 📂 Project Structure

ClarityML/
│
├── app.py
├── assets/
│ ├── banner1.png
│ ├── confusion_matrix.png
│ ├── roc_curve.png
│ ├── ml_report.png
│
├── sourcedata.csv
├── requirements.txt
└── README.md

👨‍💻 Author

Shashank Kumar
Interested in Machine Learning, Web Development & AI Systems
🔗 GitHub: [Profile]("https://github.com/Shashank14105")

⭐ Future Improvements

🚀 Production ML SaaS
login system
user uploads saved
prediction API
multi-model leaderboard UI
PDF report download
database integration (SQLite/Firebase)
SHAP explainability for model interpretation
Train/test split slider control
Multi-page dashboard UI
Docker deployment