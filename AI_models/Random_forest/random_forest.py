#!/usr/bin/env python3
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Parámetros
TEST_SIZE = 0.75
SEED = 42

# Directorio de salida (ajusta según tu convención)
out_dir = f"AI_models/Random_forest/{int((1-TEST_SIZE)*100)}_{int(TEST_SIZE*100)}"
os.makedirs(out_dir, exist_ok=True)

# 1) Carga y preprocesamiento
df = pd.read_csv("AI_models/Dataset/training/filtered_output_augmented.csv")

expected_columns = [
    'mqtt.msgid', 'mqtt.qos', 'mqtt.len', 'tcp.len', 'tcp.flags',
    'mqtt.hdrflags', 'mqtt.msgtype', 'mqtt.dupflag', 'mqtt.kalive',
    'mqtt.retain', 'tcp.time_delta', 'mqtt.conflag.cleansess',
    'mqtt.proto_len', 'mqtt.ver', 'target'
]
df = df[expected_columns]

def convert_hex_columns(df, columns):
    for col in columns:
        df[col] = df[col].apply(
            lambda x: int(x, 16) if isinstance(x, str) and x.startswith("0x") else int(x)
        )
    return df

hex_columns = ['tcp.flags', 'mqtt.hdrflags']
df = convert_hex_columns(df, hex_columns)

# 2) Separar X e y
X = df.drop("target", axis=1)
y = df["target"]

# 3) Train/test split
X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

# 4) Escalado de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 5) Codificar etiquetas
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test  = le.transform(y_test_raw)

# 6) Entrenar Random Forest
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    n_jobs=-1,
    random_state=SEED
)
start_time = time.perf_counter()
clf.fit(X_train_scaled, y_train)
end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"Entrenamiento completado en {elapsed:.2f} segundos.\n")
# 7) Evaluación
y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n=== Evaluación en Test Set ===")
print(f"Accuracy: {acc:.4f}\n")

print("=== Classification Report ===")
print(classification_report(
    y_test, y_pred, target_names=le.classes_
))

# 8) Matriz de Confusión
cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
print("=== Matriz de Confusión (raw counts) ===")
print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots()
disp.plot(ax=ax, cmap="Blues")
ax.set_title("Matriz de Confusión")
fig.tight_layout()
conf_png = os.path.join(out_dir, f"training_confusion_matrix_{int((1-TEST_SIZE)*100)}_{int(TEST_SIZE*100)}.png")
fig.savefig(conf_png)
plt.close(fig)
print(f"Matriz de confusión guardada en: {conf_png}")

# 9) Guardar artefactos
# - modelo
model_pkl = os.path.join(out_dir, f"random_forest_{int((1-TEST_SIZE)*100)}_{int(TEST_SIZE*100)}.pkl")
joblib.dump(clf, model_pkl)
# - scaler
scaler_pkl = os.path.join(out_dir, "scaler.pkl")
joblib.dump(scaler, scaler_pkl)
# - label encoder
le_pkl = os.path.join(out_dir, "label_encoder.pkl")
joblib.dump(le, le_pkl)

print("\nEntrenamiento completado.")
print("Artefactos guardados en:", out_dir)
print("  - Modelo:", model_pkl)
print("  - Scaler:", scaler_pkl)
print("  - LabelEncoder:", le_pkl)
