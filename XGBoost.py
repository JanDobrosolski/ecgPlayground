import os
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    abnormalPath = os.path.join('dataset', 'ptbdb_abnormal.csv')
    normalPath = os.path.join('dataset', 'ptbdb_normal.csv')

    abnormalData = pd.read_csv(abnormalPath, delimiter=',')
    normalData = pd.read_csv(normalPath, delimiter=',')

    abnormalData.drop(abnormalData.columns[-1], axis=1, inplace=True)
    normalData.drop(normalData.columns[-1], axis=1, inplace=True)

    abnormalData['label'] = 1
    normalData['label'] = 0

    data = pd.concat([abnormalData, normalData], axis=0)

    data = data.sample(frac=1).reset_index(drop=True)

    X = data.drop('label', axis=1)
    y = data['label']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)])

    predictions = model.predict(X_test)

    os.makedirs("results", exist_ok=True)

    xgbResultPath = os.path.join("results", "xgb")
    os.makedirs(xgbResultPath, exist_ok=True)

    cm = confusion_matrix(y_test, predictions)
    TN, FP, FN, TP = cm.ravel()

    metrics = {
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "F1_score": f1_score(y_test, predictions),
        "jaccard_score": jaccard_score(y_test, predictions),
    }

    with open(os.path.join(xgbResultPath, "metrics.json"), 'w') as file:
        json.dump(metrics, file, indent=4)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['normal', 'abnormal'], yticklabels=['normal', 'abnormal'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(xgbResultPath, "confusionMatrix.png"))
