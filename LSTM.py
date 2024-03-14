import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
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

    x = data.drop('label', axis=1)
    y = data['label']

    return x, y

def reshapeData(x):
    return np.array(x).reshape((x.shape[0], x.shape[1], -1))

def getModel(input_shape):
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(256, return_sequences=True),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model


if __name__ == "__main__":
    x, y = loadData()

    X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    X_train = reshapeData(X_train)
    X_val = reshapeData(X_val)
    X_test = reshapeData(X_test)

    model = getModel((X_train.shape[1], X_train.shape[2]))

    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tensorboard_callback = TensorBoard(log_dir="logs", histogram_freq=1)

    model.summary()

    history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping, tensorboard_callback])

    predictions = model.predict(X_test).round()

    lstmResultPath = os.path.join("results", "lstm")

    os.makedirs("results", exist_ok=True)
    os.makedirs(lstmResultPath, exist_ok=True)

    cm = confusion_matrix(y_test, predictions)
    TN, FP, FN, TP = cm.ravel()

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

    with open(os.path.join(lstmResultPath, "metrics.json"), 'w') as file:
        json.dump(metrics, file, indent=4)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['normal', 'abnormal'], yticklabels=['normal', 'abnormal'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - LSTM')
    plt.savefig(os.path.join(lstmResultPath, "confusionMatrix.png"))

    model.save(os.path.join(lstmResultPath, 'lstm_model.h5'))
