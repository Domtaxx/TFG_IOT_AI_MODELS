#!/usr/bin/env python3
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib


training_size = 0.25
SEED = 42
out_dir = f'AI_models/cnn_neural_network/{int(training_size*100)}'
os.makedirs(out_dir, exist_ok=True)

# GPU configuration (optional)
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def preprocess_data(df):
    df = df.copy()
    # Don't convert 'target' to category
    cat_columns = df.drop(columns=['target']).select_dtypes(include=['object', 'category']).columns
    df[cat_columns] = df[cat_columns].astype('category')
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df

# 1) Load & preprocess data
df = pd.read_csv("AI_models/Dataset/training/train.csv")
df = df.sample(frac=training_size, random_state=SEED).reset_index(drop=True)
df = preprocess_data(df)

dftest = pd.read_csv("AI_models/Dataset/training/test.csv")
dftest = preprocess_data(dftest)

expected_columns = [
    'mqtt.msgid', 'mqtt.qos', 'mqtt.len', 'tcp.len', 'tcp.flags',
    'mqtt.hdrflags', 'mqtt.msgtype', 'mqtt.dupflag', 'mqtt.kalive',
    'mqtt.retain', 'tcp.time_delta', 'mqtt.conflag.cleansess',
    'mqtt.proto_len', 'mqtt.conack.val', 'mqtt.conflag.passwd', 
    'mqtt.conflag.uname', 'mqtt.ver', 'mqtt.conflags','mqtt.msg', 
    'mqtt.protoname', 'target'
]
df = df[expected_columns]
dftest = dftest[expected_columns]
# Separate features and target
print(f'training size: {training_size}')
label_encoder = LabelEncoder()
Y_train_enc = label_encoder.fit_transform(df['target'].values)
Y_test_enc = label_encoder.transform(dftest['target'].values)

X_train= df.drop('target', axis=1).values
X_test = dftest.drop('target', axis=1).values

# 5) Encode labels
y_train = to_categorical(Y_train_enc)
y_test  = to_categorical(Y_test_enc)

# 6) Build 1D CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(4, activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
x_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

early = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto', restore_best_weights=True)

# 7) Train with timing
print("\nStarting CNN training...")
start_time = time.perf_counter()
history = model.fit(
    x_train_cnn, y_train,
    validation_data=(x_test_cnn, y_test),
    epochs=30,
    batch_size=64,
    callbacks=[early]
)
end_time = time.perf_counter()
print(f"\nCNN {int(training_size*100)} training completed in {end_time - start_time:.2f} seconds.\n")

# 8) Evaluate
loss, acc = model.evaluate(x_test_cnn, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")
y_pred_enc = model.predict(x_test_cnn).argmax(axis=1)
print("\nClassification Report:")
print(classification_report(Y_test_enc, y_pred_enc, target_names=label_encoder.classes_))

# 9) Confusion Matrix
cm = confusion_matrix(Y_test_enc, y_pred_enc, labels=range(len(label_encoder.classes_)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
print(pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_))
fig, ax = plt.subplots()
disp.plot(ax=ax, cmap='Blues')
ax.set_title("Confusion Matrix")
fig.tight_layout()
conf_png = os.path.join(out_dir, f'cnn_{int(training_size*100)}_confusion_matrix_training.png')
fig.savefig(conf_png)
plt.close(fig)
print(f"Confusion matrix saved to {conf_png}")

# 10) Save artifacts
le_path     = os.path.join(out_dir, 'cnn_label_encoder.pkl')
tflite_path = os.path.join(out_dir, 'cnn_model.tflite')

joblib.dump(label_encoder, le_path)
print("Saved Keras CNN model and preprocessors.")

# 11) Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"Saved TensorFlow Lite CNN model to {tflite_path}")
