import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import emlearn
import matplotlib.pyplot as plt
import joblib


# 1) Carga y preprocesa tus datos
df = pd.read_csv('AI_models/Dataset/training/filtered_output_augmented.csv')  # ajusta la ruta si hace falta
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
def convert_hex_columns(df, columns):
    for col in columns:
        df[col] = df[col].apply(
            lambda x: int(x, 16) if isinstance(x, str) and x.startswith('0x') else 0
        )
    return df
hex_columns = ['tcp.flags', 'mqtt.hdrflags']
df = df[expected_columns]
df = convert_hex_columns(df, hex_columns)
X = df.drop('target', axis=1)
y = df['target']

# 2) Divide en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3) Entrena el modelo Random Forest usando todos los núcleos
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    n_jobs=-1,           # paraleliza en todos los cores
    random_state=42
)
clf.fit(X_train, y_train)

# 4) Evaluación en el set de prueba
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n=== Evaluación en Test Set ===")
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5) Matriz de confusión
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
print("Matriz de Confusión:")
print(cm)

# 6) Display gráfico de la matriz
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
fig, ax = plt.subplots()
disp.plot(ax=ax, cmap='Blues')
ax.set_title("Matriz de Confusión")
fig.tight_layout()

# aquí guardas la figura en PNG
fig.savefig('AI_models/Random_forest/70_30/confusion_matrix_training_70_30.png')
plt.close(fig)

# 7) Convierte a TinyML (código C inline)
joblib.dump(clf, "AI_models/Random_forest/70_30/random_forest_70_30.pkl")

print("\nEntrenamiento completado y modelo TinyML generado en random_forest_70_30")