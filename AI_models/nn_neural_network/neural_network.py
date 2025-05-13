import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
model_training_porcentage = 0.7


# 0) GPU setup: list devices & enable memory growth
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 1) Load & preprocess data
df = pd.read_csv('AI_models/Dataset/training/filtered_output_augmented.csv')
# — drop any text cols if present —
for col in ['mqtt.msg','mqtt.protoname','mqtt.willtopic','mqtt.topic']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)
# — convert hex cols to int —
hex_cols = ['tcp.flags','mqtt.conflags','mqtt.conack.flags','mqtt.hdrflags']
for col in hex_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: int(x, 0) if isinstance(x, str) else int(x))

expected_columns = [
    'tcp.flags',
    'tcp.time_delta',
    'tcp.len',
    'mqtt.conflag.cleansess',
    'mqtt.conflag.qos',
    'mqtt.conflag.retain',
    'mqtt.dupflag',
    'mqtt.kalive',
    'mqtt.len',
    'mqtt.msgtype',
    'mqtt.qos',
    'mqtt.retain',
    #'mqtt.topic',
    'mqtt.hdrflags',
    'mqtt.conflag.willflag',
    'mqtt.sub.qos',
    'mqtt.suback.qos',
    'mqtt.conack.flags.sp',
    'target'
]
df = df[expected_columns]
X = df.drop('target', axis=1).values
y = df['target'].values

# 2) Split, scale, encode
X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=model_training_porcentage, random_state=42
)
print(f'test size:{model_training_porcentage}, training size: {1-model_training_porcentage}')
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
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(y_train.shape[1], activation='softmax'),
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4) Train (on GPU automatically)
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)

# 5) Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.4f}")

# 6) Classification report
y_pred_enc = model.predict(X_test).argmax(axis=1)
print("\nClassification Report:")
print(classification_report(y_test_enc, y_pred_enc, target_names=le.classes_))

# 7) Confusion matrix (console + PNG)
cm = confusion_matrix(y_test_enc, y_pred_enc, labels=range(len(le.classes_)))
print("\nConfusion Matrix (raw counts):")
print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots()
disp.plot(ax=ax, cmap='Blues')   # white→blue gradient
ax.set_title("Confusion Matrix")
fig.tight_layout()
fig.savefig('AI_models/nn_neural_network/30_70/nn_confusion_matrix_training.png')
plt.close(fig)
print("\nSaved confusion matrix to AI_models/nn_neural_network/30_70/nn_confusion_matrix.png")

# 8) Save artifacts
import joblib
joblib.dump(scaler, 'AI_models/nn_neural_network/30_70/nn_scaler.pkl')
joblib.dump(le, 'AI_models/nn_neural_network/30_70/nn_label_encoder.pkl')
model.save('AI_models/nn_neural_network/30_70/nn_model.h5')

# 9) Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('AI_models/nn_neural_network/30_70/nn_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Saved TensorFlow Lite model to AI_models/nn_neural_network/30_70/nn_model.tflite")
