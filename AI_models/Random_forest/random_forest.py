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
from sklearn.tree import DecisionTreeClassifier

# Parámetros
training_size = 0.25
SEED = 42
def preprocess_data(df):
    df = df.copy()
    # Don't convert 'target' to category
    cat_columns = df.drop(columns=['target']).select_dtypes(include=['object', 'category']).columns
    df[cat_columns] = df[cat_columns].astype('category')
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df

# Directorio de salida (ajusta según tu convención)
out_dir = f"AI_models/Random_forest/{int(training_size*100)}"
os.makedirs(out_dir, exist_ok=True)

# 1) Carga y preprocesamiento
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
print("Training classes:", df['target'].unique())
print("Test classes:", dftest['target'].unique())
# Ajustar el encoder con los datos de entrenamiento
label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(df['target'].values)
Y_test = label_encoder.transform(dftest['target'].values)

X_train = df.drop('target', axis=1).values
X_test = dftest.drop('target', axis=1).values

# 6) Entrenar Random Forest
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=SEED
)

start_time = time.perf_counter()
clf.fit(X_train, Y_train)
end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"Entrenamiento completado en {elapsed:.2f} segundos.\n")
# 7) Evaluación
y_pred = clf.predict(X_test)
acc = accuracy_score(Y_test, y_pred)
print(f"\n=== Evaluación en Test Set ===")
print(f"Accuracy: {acc:.4f}\n")

print("=== Classification Report ===")
print(classification_report(
    label_encoder.inverse_transform(Y_test),
    label_encoder.inverse_transform(y_pred)
))

# 8) Matriz de Confusión
class_labels = label_encoder.classes_
cm = confusion_matrix(
    label_encoder.inverse_transform(Y_test),
    label_encoder.inverse_transform(y_pred),
    labels=class_labels
)
print("=== Matriz de Confusión (raw counts) ===")
print(pd.DataFrame(cm))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
fig, ax = plt.subplots()
disp.plot(ax=ax, cmap="Blues")
ax.set_title("Matriz de Confusión")
fig.tight_layout()
conf_png = os.path.join(out_dir, f"rf_{int(training_size*100)}_training_confusion_matrix.png")
fig.savefig(conf_png)
plt.close(fig)
print(f"Matriz de confusión guardada en: {conf_png}")

# 9) Guardar artefactos
# - modelo
model_pkl = os.path.join(out_dir, f"random_forest.pkl")
joblib.dump(clf, model_pkl)

encoder_pkl = os.path.join(out_dir, "label_encoder.pkl")
joblib.dump(label_encoder, encoder_pkl)


print("\nEntrenamiento completado.")
print("Artefactos guardados en:", out_dir)
print("  - Modelo:", model_pkl)
print("  - LabelEncoder:", encoder_pkl)

