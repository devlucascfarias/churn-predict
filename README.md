# Churn Predict

A Machine Learning application to predict customer **Churn** (cancellation probability). Includes an interactive [Streamlit](https://streamlit.io/) dashboard and a training script.

## Features

- **Real-Time Prediction:** Input customer data and get instant churn probabilities.
- **Dataset Insights:** Visualize distributions, correlations, and customer behavior.
- **Model Report:** Performance metrics and robustness tests.
- **Automated Training:** Script for preprocessing, training (Random Forest), and evaluation.

## Tech Stack

- **Python 3**
- **Streamlit** (Web Dashboard)
- **Scikit-learn** (Modeling)
- **Pandas & NumPy** (Data Manipulation)
- **Plotly, Matplotlib & Seaborn** (Visualization)

### 1. Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn joblib
```

### 2. Run Dashboard

```bash
streamlit run app.py
```

### 3. Train Model (Optional)

```bash
python churn_model.py
```

## Structure

- `app.py`: Main Streamlit dashboard.
- `churn_model.py`: Training and preprocessing script.
- `data/`: Contains datasets, saved models (`.pkl`), and metrics.

## About the Model

Uses a **Random Forest Classifier**. Developed with a focus on generalization, addressing initial data leakage issues to achieve an accuracy of approximately **90%**.
