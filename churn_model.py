"""
Churn Modeling Script

This script handles the entire training pipeline for the Churn prediction model.
It performs:
1. Loading training and testing data.
2. Preprocessing (cleaning, encoding, and normalization).
3. Training a Random Forest Classifier.
4. Model evaluation (Accuracy, Confusion Matrix, Feature Importance).
5. Persistence of the model and auxiliary objects (scalers, encoders).
6. Generating predictions for the test set.
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler

np.random.seed(42)
sns.set_theme(style="whitegrid")

DROP_FEATURES = ['CustomerID', 'Support Calls', 'Total Spend']

def preprocess_data(df, is_train=True, label_encoders=None, scaler=None, drop_cols=None):
    """
    Performs data preprocessing (Cleaning, Encoding, Scaling).

    Args:
        df (pd.DataFrame): Original DataFrame.
        is_train (bool): If True, fits the encoders and scaler. If False, only transforms.
        label_encoders (dict, optional): Dictionary of fitted encoders (used when is_train=False).
        scaler (StandardScaler, optional): Fitted scaler (used when is_train=False).
        drop_cols (list, optional): Additional list of columns to remove.

    Returns:
        tuple or pd.DataFrame:
            - If is_train=True: Returns (df_processed, label_encoders, scaler, numeric_cols).
            - If is_train=False: Returns only df_processed.
    """
    df_processed = df.copy()
    
    if drop_cols:
        to_drop = list(set(DROP_FEATURES + drop_cols))
    else:
        to_drop = DROP_FEATURES
        
    df_processed = df_processed.drop(columns=[c for c in to_drop if c in df_processed.columns], errors='ignore')

    if is_train:
        df_processed = df_processed.dropna()
    
    categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']
    
    if is_train:
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
            
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if 'Churn' in numeric_cols: numeric_cols.remove('Churn')
        
        scaler = StandardScaler()
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
        
        return df_processed, label_encoders, scaler, numeric_cols
    else:
        for col in categorical_cols:
            le = label_encoders[col]
            df_processed[col] = df_processed[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
            df_processed[col] = le.transform(df_processed[col].astype(str))
            
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        df_processed[numeric_cols] = scaler.transform(df_processed[numeric_cols])
        
        return df_processed

if __name__ == "__main__":
    """
    Main Execution Block
    
    1. Loads datasets.
    2. Applies preprocessing.
    3. Trains Random Forest model.
    4. Evaluates performance on validation.
    5. Saves artifacts (model and transformers).
    6. Generates test submission.
    """

    train_path = 'data/customer_churn_dataset-training-master.csv'
    test_path = 'data/customer_churn_dataset-testing-master.csv'

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"Initial Shapes: Train {df_train.shape}, Test {df_test.shape}")

    df_train_proc, encoders, scaler, num_features = preprocess_data(df_train, is_train=True)

    X = df_train_proc.drop('Churn', axis=1)
    y = df_train_proc['Churn']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    y_pred_val = rf_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)

    print(f"\nValidation Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_val, y_pred_val))

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred_val, ax=ax[0], cmap='Blues')
    ax[0].set_title('Confusion Matrix')

    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)
    importances.plot(kind='barh', ax=ax[1])
    ax[1].set_title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    joblib.dump(rf_model, 'data/churn_model.pkl')
    joblib.dump(encoders, 'data/encoders.pkl')
    joblib.dump(scaler, 'data/scaler.pkl')
    
    with open('data/metrics.json', 'w') as f:
        json.dump({'accuracy': acc}, f)
    
    X_test_raw = df_test.drop('Churn', axis=1) if 'Churn' in df_test.columns else df_test
    df_test_proc = preprocess_data(X_test_raw, is_train=False, label_encoders=encoders, scaler=scaler)
    
    print("\nGenerating predictions for test file...")
    test_preds = rf_model.predict(df_test_proc)
    
    submission = pd.DataFrame({
        'CustomerID': df_test['CustomerID'] if 'CustomerID' in df_test.columns else range(len(test_preds)),
        'Churn_Prediction': test_preds
    })
    submission.to_csv('data/churn_predictions.csv', index=False)
    print("Process completed successfully!")