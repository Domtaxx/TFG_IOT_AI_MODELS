import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 🔹 Hex to ASCII converter
def hex_to_ascii_safe(h):
    try:
        return bytes.fromhex(str(h)).decode('utf-8')
    except Exception:
        return ""
def plot_confusion_matrix(y_true, y_pred, labels, filename="results/confusion_matrix_random_forest.png"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"✅ Confusion matrix saved as {filename}")

def convert_hex_columns(df, columns):
    for col in columns:
        df[col] = df[col].apply(
            lambda x: int(x, 16) if isinstance(x, str) and x.startswith('0x') else 0
        )
    return df

hex_columns = ['tcp.flags', 'mqtt.hdrflags']
# 🔹 Main evaluation logic
def evaluate_model(input_csv, input_csv2, output_csv, show_plot=False):
    # Load trained components
    model = joblib.load("AI_models/Random_forest/random_forest_model.pkl")
    scaler = joblib.load("AI_models/Random_forest/scaler.pkl")
    label_encoder = joblib.load("AI_models/Random_forest/label_encoder.pkl")
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
    # Load real-world captured dataset
    df = pd.read_csv(input_csv, header=0)
    df = df.rename(columns={'Line_origin': 'target'})
    df = df[expected_columns]
    df2 = pd.read_csv(input_csv2, header=0)
    df2 = df2[expected_columns]

    df = pd.concat([df, df2], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.drop(df[df['target'] == 'UNKNOWN'].index, inplace=True)
    df.drop(df[df['target'] == 'ddos'].index, inplace=True)
    df = convert_hex_columns(df, hex_columns)
    # Convert hex fields if needed
    #df["mqtt.msg"] = df["mqtt.msg"].replace(0.0, "").fillna("").astype(str)
    #df["mqtt.msg"] = df["mqtt.msg"].apply(hex_to_ascii_safe)

    # Feature preparation
    feature_names = list(model.feature_names_in_)
    df = df.fillna(0.0)
    df = df.replace({'True': 1, 'False': 0})
    X = df[feature_names].copy()

    # Scale features
    X_scaled = scaler.transform(X)

    # Prediction
    y_pred = model.predict(X_scaled)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    df["predicted"] = y_pred_labels

    # Detect anomalies
    df["is_anomaly"] = df["predicted"] != "legitimate"

    # Evaluation
    if "target" in df.columns:
        print("\n📊 Classification Report:")
        print(classification_report(df["target"], df["predicted"]))

        acc = accuracy_score(df["target"], df["predicted"])
        print(f"\n✅ Accuracy: {acc:.4f}")

        if show_plot:
            print("\n🖼 Showing Confusion Matrix")
            labels = sorted(label_encoder.classes_)
            plot_confusion_matrix(df["target"], df["predicted"], labels=labels)

    # Save results
    df.to_csv(output_csv, index=False, na_rep="")
    print(f"\n✅ Predictions saved to {output_csv}")

evaluate_model("predict_attacks_real_time/raw_data.csv", "attack_data.csv", "results/output.csv", show_plot=True)