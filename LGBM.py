import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
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

    abnormalData.columns = np.arange(len(abnormalData.columns))
    normalData.columns = np.arange(len(normalData.columns))

    abnormalData['label'] = 1
    normalData['label'] = 0

    data = pd.concat([abnormalData, normalData], axis=0)

    data = data.sample(frac=1).reset_index(drop=True)

    x = data.drop('label', axis=1)
    y = data['label']

    X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_data = lgbm.Dataset(X_train, label=y_train)
    valid_data = lgbm.Dataset(X_val, label=y_val)

    parameters = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'is_unbalance': 'true',
        'boosting': 'gbdt',
        'num_leaves': 64,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 20,
        'learning_rate': 0.05,
        'verbose': 1
    }

    model = lgbm.train(parameters,
                      train_data,
                      valid_sets=valid_data,
                      num_boost_round=500)

    predictions = model.predict(X_test)
    predictions = (predictions >= 0.5).astype(int)

    os.makedirs("results", exist_ok=True)

    lgbmResultPath = os.path.join("results", "lgbm")
    os.makedirs(lgbmResultPath, exist_ok=True)

    cm = confusion_matrix(y_test, predictions)
    TN, FP, FN, TP = cm.ravel()

    metrics = {
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
        "accuracy": round(accuracy_score(y_test, predictions)*100, 2),
        "precision": round(precision_score(y_test, predictions)*100, 2),
        "recall": round(recall_score(y_test, predictions), 2)*100,
        "F1_score": round(f1_score(y_test, predictions), 2)*100,
        "jaccard_score": round(jaccard_score(y_test, predictions)*100, 2),
    }

    with open(os.path.join(lgbmResultPath, "metrics.json"), 'w') as file:
        json.dump(metrics, file, indent=4)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['normal', 'abnormal'], yticklabels=['normal', 'abnormal'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(lgbmResultPath, "confusionMatrix.png"))
