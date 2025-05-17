#!/usr/bin/env python3
import os
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# Proporción de test/training
TEST_SIZE = 0.05
SEED = 42

# 0) GPU setup: list devices & enable memory growth
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
X = df.drop('target', axis=1).values
y = df['target'].values

# 2) Split, scale, encode
X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=TEST_SIZE, train_size=1-TEST_SIZE, random_state=SEED,stratify=y
)
print(f'test size: {TEST_SIZE}, training size: {1-TEST_SIZE}')
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_raw)
y_test_enc  = le.transform(y_test_raw)
y_train = to_categorical(y_train_enc)
y_test  = to_categorical(y_test_enc)

# 3) Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dense(y_train.shape[1], activation='softmax'),
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#Prepare callbacks
early = EarlyStopping(
    monitor='val_loss',
    patience=5,               # stop if no val_loss improvement for 5 epochs
    min_delta=1e-3,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,               # drop LR by half
    patience=3,               # if no val_loss improvement for 3 epochs
    min_lr=1e-6
)


# 4) Train (on GPU automatically) with timing
print("\nStarting training...")
start_time = time.perf_counter()
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early, reduce_lr]
)
end_time = time.perf_counter()
print(f"\nNN {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} Training completed in {end_time - start_time:.2f} seconds.\n")

# 5) Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}\n")

# 6) Classification report
y_pred_enc = model.predict(X_test).argmax(axis=1)
print("Classification Report:")
print(classification_report(y_test_enc, y_pred_enc, target_names=le.classes_))

# 7) Confusion matrix (console + PNG)
cm = confusion_matrix(y_test_enc, y_pred_enc, labels=range(len(le.classes_)))
print("\nConfusion Matrix (raw counts):")
print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots()
disp.plot(ax=ax, cmap='Blues')
ax.set_title("Confusion Matrix")
fig.tight_layout()
out_dir = f'AI_models/nn_neural_network/{int((1-TEST_SIZE)*100)}_{int((TEST_SIZE)*100)}'
os.makedirs(out_dir, exist_ok=True)
cm_path = os.path.join(out_dir, 'nn_confusion_matrix_training.png')
fig.savefig(cm_path)
plt.close(fig)
print(f"Saved confusion matrix to {cm_path}\n")

# 8) Save artifacts
scale_path = os.path.join(out_dir, 'nn_scaler.pkl')
le_path    = os.path.join(out_dir, 'nn_label_encoder.pkl')
model_h5   = os.path.join(out_dir, 'nn_model.h5')
model_tflite = os.path.join(out_dir, 'nn_model.tflite')

joblib.dump(scaler, scale_path)
joblib.dump(le, le_path)
model.save(model_h5)
print("Saved scaler, label encoder, and Keras model.")

# 9) Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(model_tflite, 'wb') as f:
    f.write(tflite_model)
print(f"Saved TensorFlow Lite model to {model_tflite}")
