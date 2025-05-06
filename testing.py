import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 🔹 Optional: Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# 🔹 Hex to ASCII converter
def hex_to_ascii_safe(h):
    try:
        return bytes.fromhex(str(h)).decode('utf-8')
    except Exception:
        return ""

# 🔹 Main evaluation logic
def evaluate_model(input_csv, input_csv2, output_csv, show_plot=False):
    # Load trained components
    model = joblib.load("AI_models/Random_forest/random_forest_model.pkl")
    scaler = joblib.load("AI_models/Random_forest/scaler.pkl")
    label_encoder = joblib.load("AI_models/Random_forest/label_encoder.pkl")
    expected_columns = [
        'tcp.time_delta', 
        'tcp.len', 
        'mqtt.conack.flags.reserved', 
        'mqtt.conack.flags.sp', 
        'mqtt.conack.val', 
        'mqtt.conflag.cleansess', 
        'mqtt.conflag.passwd', 
        'mqtt.conflag.qos', 
        'mqtt.conflag.reserved', 
        'mqtt.conflag.retain', 
        'mqtt.conflag.uname', 
        'mqtt.conflag.willflag', 
        'mqtt.dupflag', 
        'mqtt.kalive', 
        'mqtt.len', 
        'mqtt.msgid', 
        'mqtt.msgtype', 
        'mqtt.proto_len', 
        'mqtt.qos', 
        'mqtt.retain', 
        'mqtt.sub.qos', 
        'mqtt.suback.qos', 
        'mqtt.ver', 
        'mqtt.willmsg', 
        'mqtt.willmsg_len', 
        'mqtt.willtopic', 
        'mqtt.willtopic_len',
        'target'
    ]

    # Load real-world captured dataset
    df = pd.read_csv(input_csv)
    df = df.rename(columns={'Line_origin': 'target'})
    df = df[expected_columns]
    df2 = pd.read_csv(input_csv2)
    df2 = df2[expected_columns]

    df = pd.concat([df, df2], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.drop(df[df['target'] == 'UNKNOWN'].index, inplace=True)
    df.drop(df[df['target'] == 'dos'].index, inplace=True)

    # Convert hex fields if needed
    #df["mqtt.msg"] = df["mqtt.msg"].replace(0.0, "").fillna("").astype(str)
    #df["mqtt.msg"] = df["mqtt.msg"].apply(hex_to_ascii_safe)

    # Feature preparation
    feature_names = list(model.feature_names_in_)
    df = df.fillna(0.0)
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
            labels = sorted(df["target"].unique())
            plot_confusion_matrix(df["target"], df["predicted"], labels=labels)

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Predictions saved to {output_csv}")

evaluate_model("predict_attacks_real_time/raw_data.csv", "attack_traffic_log.csv", "output.csv", show_plot=True)