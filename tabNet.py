import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from tabnet import TabNetClassifier
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import confusion_matrix

def loadData():
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
    return data

if __name__ == "__main__":
    data = loadData()

    x = data.drop('label', axis=1)
    y = np.array([[1, 0] if label == 0 else [0, 1] for label in data['label']])

    X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert features into a dictionary format
    feature_dict_train = {str(i): X_train.iloc[:, i] for i in range(X_train.shape[1])}
    feature_dict_val = {str(i): X_val.iloc[:, i] for i in range(X_val.shape[1])}
    feature_dict_test = {str(i): X_test.iloc[:, i] for i in range(X_test.shape[1])}

    # Convert labels into a dictionary format
    label_dict_train = {'output_1': y_train}
    label_dict_val = {'output_1': y_val}
    label_dict_test = {'output_1': y_test}

    # Define TensorFlow feature columns
    feature_columns = [tf.feature_column.numeric_column(str(i)) for i in range(X_train.shape[1])]

    model = TabNetClassifier(feature_columns=feature_columns, num_classes=2, feature_dim=64, output_dim=32)

    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tensorboard_callback = TensorBoard(log_dir="logs", histogram_freq=1)

    model.fit(feature_dict_train, label_dict_train, epochs=100, batch_size=32, validation_data=(feature_dict_val, label_dict_val), callbacks=[early_stopping,tensorboard_callback])

    preds = np.argmax(model.predict(feature_dict_test), axis=-1)
    gt = np.argmax(y_test,axis=-1)

    cm = confusion_matrix(gt,preds)
    TN, FP, FN, TP = cm.ravel()

    tabNetResultPath = os.path.join("results", "tabNet")

    os.makedirs("results", exist_ok=True)
    os.makedirs(tabNetResultPath, exist_ok=True)

    metrics = {
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
        "accuracy": round((TP + TN) / (TP + TN + FP + FN), 2),
        "precision": round(TP / (TP + FP), 2),
        "recall": round(TP / (TP + FN), 2),
        "F1_score": round(2 * TP / (2 * TP + FP + FN), 2)
    }

    with open(os.path.join(tabNetResultPath, "metrics.json"), 'w') as file:
        json.dump(metrics, file, indent=4)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['normal', 'abnormal'], yticklabels=['normal', 'abnormal'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - LSTM')
    plt.savefig(os.path.join(tabNetResultPath, "confusionMatrix.png"))
