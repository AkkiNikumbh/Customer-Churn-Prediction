# 📊 Customer Churn Prediction & Retention Analytics

An end-to-end Machine Learning pipeline and interactive Streamlit web dashboard designed to analyze telecom customer churn drivers, predict at-risk customer accounts with **85.6% Accuracy**, and generate targeted retention strategies projected to reduce annual revenue loss by 12%.

---

## 🎯 Executive Summary & Resume Highlights

- **Predictive Churn Model**: Engineered an end-to-end Random Forest ML pipeline processing **50,000+ customer data points** to classify churn risk.
- **High Model Performance**: Achieved **85.65% Model Accuracy**, **89.05% Precision**, **85.32% Recall**, and **0.9991 ROC-AUC**.
- **Interactive Analytics Dashboard**: Developed a full-featured Streamlit web application with real-time KPI metrics, exploratory visualizations, predictive simulator, and automated high-risk outreach exports.
- **Business Revenue Impact**: Proposes data-driven customer retention strategies projected to cut annual revenue churn by **12%**.

---

## 🚀 Key Features

1. **📈 Executive KPI Summary**: Real-time overview of churn rates, total revenue loss, high-risk customer counts, and model accuracy.
2. **🔍 Exploratory Data Analysis (EDA)**: Interactive charts analyzing key churn drivers such as tenure, total spend, support call volume, and contract type.
3. **🔮 Real-Time Risk Simulator**: Input hypothetical or real customer profile parameters to get immediate churn probability scores and personalized retention recommendations.
4. **🎯 At-Risk Retention Outreach Queue**: Filter high-risk and medium-risk accounts and export custom CSV files for customer success teams.

---

## 🛠️ Tech Stack & Dependencies

- **Language**: Python 3.12+
- **Machine Learning**: `scikit-learn`, `joblib`
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization & Web App**: `streamlit`, `plotly`

---

## 📁 Repository Structure

```
Customer-Churn-Prediction/
├── app.py                                       # Main Streamlit Dashboard Application
├── model_pipeline.py                            # ML Training & Evaluation Pipeline
├── dataset_generator.py                         # Synthetic Data Generation Script
├── churn_model.joblib                           # Trained Random Forest Model Artifact
├── model_metrics.json                           # Model Performance Evaluation Metrics
├── requirements.txt                             # Python Dependencies
├── customer_churn_dataset-training-master.csv   # Training Dataset
└── customer_churn_dataset-testing-master.csv    # Testing Dataset
```

---

## ⚡ How to Run Locally

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/AkkiNikumbh/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install -r requirements.txt
```

### 2. (Optional) Run ML Pipeline
To re-train the Random Forest model and regenerate metrics:
```bash
python model_pipeline.py
```

### 3. Launch Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `https://akkinikumbh-customer-churn-prediction-app-pfhymh.streamlit.app/`.

---

## 📊 Model Evaluation Results

- **Sample Size Analyzed**: 55,000 Data Points
- **Accuracy**: 85.65%
- **Precision**: 89.05%
- **Recall**: 85.32%
- **F1 Score**: 0.8715
- **ROC-AUC**: 0.9991
