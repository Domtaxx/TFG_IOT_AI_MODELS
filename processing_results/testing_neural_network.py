import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 🔹 Confusion Matrix Plot
def plot_confusion_matrix(y_true, y_pred, labels, filename="results/confusion_matrix_nerual_80.png"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"✅ Confusion matrix saved as {filename}")

# 🔹 Model architecture must match the training one
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.out = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.out(x)
def convert_hex_columns(df, columns):
    for col in columns:
        df[col] = df[col].apply(
            lambda x: int(x, 16) if isinstance(x, str) and x.startswith('0x') else 0
        )
    return df

hex_columns = ['tcp.flags', 'mqtt.hdrflags']
# 🔹 Main evaluation logic
def evaluate_model(input_csv, input_csv2, output_csv, show_plot=False):
    expected_columns = [
        'tcp.time_delta','tcp.flags', 'tcp.len', 'mqtt.conack.flags.reserved', 'mqtt.conack.flags.sp',
        'mqtt.conack.val', 'mqtt.conflag.cleansess', 'mqtt.conflag.passwd', 'mqtt.conflag.qos',
        'mqtt.conflag.reserved', 'mqtt.conflag.retain', 'mqtt.conflag.uname', 'mqtt.conflag.willflag',
        'mqtt.dupflag', 'mqtt.kalive', 'mqtt.len', 'mqtt.msgid', 'mqtt.msgtype', 'mqtt.proto_len',
        'mqtt.qos', 'mqtt.retain', 'mqtt.sub.qos', 'mqtt.suback.qos', 'mqtt.ver', 'mqtt.willmsg',
        'mqtt.willmsg_len', 'mqtt.willtopic', 'mqtt.willtopic_len', 'mqtt.hdrflags', 'target'
    ]

    # 🔸 Load model & preprocessing objects
    scaler = joblib.load("AI_models/neural_network/scaler_80_20.pkl")
    label_encoder = joblib.load("AI_models/neural_network/label_encoder_80_20.pkl")
    model = NeuralNet(input_size=29, num_classes=len(label_encoder.classes_))
    model.load_state_dict(torch.load("AI_models/neural_network/modelo_mqtt_80_20.pt"))
    model.eval()

    # 🔸 Load and combine datasets
    df = pd.read_csv(input_csv)
    df = df.rename(columns={'Line_origin': 'target'})
    df = df[expected_columns]
    df2 = pd.read_csv(input_csv2)
    df2 = df2[expected_columns]

    df = pd.concat([df, df2], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.drop(df[df['target'] == 'UNKNOWN'].index, inplace=True)
    df.drop(df[df['target'] == 'ddos'].index, inplace=True)  

    df = df.fillna(0.0)

    # 🔸 Prepare features
    df = df[expected_columns]
    df = df.replace({'True': 1, 'False': 0})
    df = convert_hex_columns(df, hex_columns)
    X = df.drop("target", axis=1)
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # 🔸 Predict
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs, dim=1)
        predicted_labels = label_encoder.inverse_transform(predictions.numpy())

    df["predicted"] = predicted_labels
    df["is_anomaly"] = df["predicted"] != "legitimate"

    # 🔸 Evaluation
    if "target" in df.columns:
        print("\n📊 Classification Report:")
        print(classification_report(df["target"], df["predicted"]))

        acc = accuracy_score(df["target"], df["predicted"])
        print(f"\n✅ Accuracy: {acc:.4f}")

        if show_plot:
            print("\n🖼 Showing Confusion Matrix")
            labels = sorted(df["target"].unique())
            plot_confusion_matrix(df["target"], df["predicted"], labels=labels)

    # 🔸 Save results
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Predictions saved to {output_csv}")

# Example usage
evaluate_model("predict_attacks_real_time/raw_data.csv", "malformed_tshark_labeled.csv", "results/output_nn.csv", show_plot=True)
