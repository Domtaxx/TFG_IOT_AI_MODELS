#!/usr/bin/env python3
import os
import time
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# Proporción de test/training
training_size = 0.25
SEED = 42

# 0) GPU setup: list devices & enable memory growth
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

label_encoder = LabelEncoder()
Y_train_enc = label_encoder.fit_transform(df['target'].values)
Y_test_enc = label_encoder.transform(dftest['target'].values)

X_train= df.drop('target', axis=1).values
X_test = dftest.drop('target', axis=1).values

print(f'training size: {training_size}')
Y_train = to_categorical(Y_train_enc)
Y_test  = to_categorical(Y_test_enc)

# 3) Build model
model = Sequential([
    Dense(50, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(30, activation='relu'),
    Dense(20, activation='relu'),
    Dense(Y_train.shape[1], activation='softmax'),#y_train.shape[1] = 4
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Prepare callbacks
early = EarlyStopping(
    monitor='val_loss',
    patience=5,               # stop if no val_loss improvement for 5 epochs
    min_delta=1e-3,
    verbose=1, 
    mode='auto',
    restore_best_weights=True
)


# 4) Train (on GPU automatically) with timing
print("\nNN Starting training...")
start_time = time.perf_counter()
history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=200,
    batch_size=64,
    callbacks=[early]
)
end_time = time.perf_counter()
print(f"\nNN {int(training_size*100)}% Training completed in {end_time - start_time:.2f} seconds.\n")

# 5) Evaluate
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}\n")

# 6) Classification report
y_pred_enc = model.predict(X_test).argmax(axis=1)
print("Classification Report:")
print(classification_report(Y_test_enc, y_pred_enc, target_names=label_encoder.classes_))

# 7) Confusion matrix (console + PNG)
cm = confusion_matrix(Y_test_enc, y_pred_enc, labels=range(len(label_encoder.classes_)))
print("\nConfusion Matrix (raw counts):")
print(pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
fig, ax = plt.subplots()
disp.plot(ax=ax, cmap='Blues')
ax.set_title("Confusion Matrix")
fig.tight_layout()
out_dir = f'AI_models/nn_neural_network/{int((training_size)*100)}'
os.makedirs(out_dir, exist_ok=True)
cm_path = os.path.join(out_dir, f'nn_{int(training_size*100)}_confusion_matrix_training.png')
fig.savefig(cm_path)
plt.close(fig)
print(f"Saved confusion matrix to {cm_path}\n")

# 8) Save artifacts
le_path    = os.path.join(out_dir, 'nn_label_encoder.pkl')
model_tflite = os.path.join(out_dir, 'nn_model.tflite')

joblib.dump(label_encoder, le_path)
# 9) Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(model_tflite, 'wb') as f:
    f.write(tflite_model)
print(f"Saved TensorFlow Lite model to {model_tflite}")
