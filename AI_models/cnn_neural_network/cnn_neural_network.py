#!/usr/bin/env python3
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib


TEST_SIZE = 0.75
out_dir = f'AI_models/cnn_neural_network/{int((1-TEST_SIZE)*100)}_{int(TEST_SIZE*100)}'
os.makedirs(out_dir, exist_ok=True)

# GPU configuration (optional)
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 1) Load & preprocess data
df = pd.read_csv('AI_models/Dataset/training/filtered_output_augmented.csv')
for col in ['mqtt.msg','mqtt.protoname','mqtt.willtopic','mqtt.topic']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)
hex_cols = ['tcp.flags','mqtt.conflags','mqtt.conack.flags','mqtt.hdrflags']
for col in hex_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: int(x, 0) if isinstance(x, str) else int(x))

expected_columns = [
    'mqtt.msgid', 'mqtt.qos', 'mqtt.len', 'tcp.len', 'tcp.flags',
    'mqtt.hdrflags', 'mqtt.msgtype', 'mqtt.dupflag', 'mqtt.kalive',
    'mqtt.retain', 'tcp.time_delta', 'mqtt.conflag.cleansess',
    'mqtt.proto_len', 'mqtt.ver','target'
]
df = df[expected_columns]

# Separate features and target
FEATURES = [c for c in df.columns if c != 'target']
X = df[FEATURES].values
y = df['target'].values

# 2) Train/test split
X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42
)
print(f'test size: {TEST_SIZE}, training size: {1-TEST_SIZE}')

# 3) Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 4) Reshape for Conv1D
X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
X_test_cnn  = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

# 5) Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_raw)
y_test_enc  = le.transform(y_test_raw)
y_train = to_categorical(y_train_enc)
y_test  = to_categorical(y_test_enc)

# 6) Build 1D CNN model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_train.shape[1], activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 7) Train with timing
print("\nStarting CNN training...")
start_time = time.perf_counter()
history = model.fit(
    X_train_cnn, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)
end_time = time.perf_counter()
print(f"\nCNN {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} training completed in {end_time - start_time:.2f} seconds.\n")

# 8) Evaluate
loss, acc = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")
y_pred_enc = model.predict(X_test_cnn).argmax(axis=1)
print("\nClassification Report:")
print(classification_report(y_test_enc, y_pred_enc, target_names=le.classes_))

# 9) Confusion Matrix
cm = confusion_matrix(y_test_enc, y_pred_enc, labels=range(len(le.classes_)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))
fig, ax = plt.subplots()
disp.plot(ax=ax, cmap='Blues')
ax.set_title("Confusion Matrix (1D CNN)")
fig.tight_layout()
conf_png = os.path.join(out_dir, 'cnn_confusion_matrix_training.png')
fig.savefig(conf_png)
plt.close(fig)
print(f"Confusion matrix saved to {conf_png}")

# 10) Save artifacts
scaler_path = os.path.join(out_dir, 'cnn_scaler.pkl')
le_path     = os.path.join(out_dir, 'cnn_label_encoder.pkl')
model_h5    = os.path.join(out_dir, 'cnn_model.h5')
tflite_path = os.path.join(out_dir, 'cnn_model.tflite')

joblib.dump(scaler, scaler_path)
joblib.dump(le, le_path)
model.save(model_h5)
print("Saved Keras CNN model and preprocessors.")

# 11) Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"Saved TensorFlow Lite CNN model to {tflite_path}")
