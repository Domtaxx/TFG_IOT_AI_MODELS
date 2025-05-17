#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
from tflite_runtime.interpreter import Interpreter
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# List of feature column names in the same order used at training
FEATURES = [
    'mqtt.msgid', 'mqtt.qos', 'mqtt.len', 'tcp.len', 'tcp.flags',
    'mqtt.hdrflags', 'mqtt.msgtype', 'mqtt.dupflag', 'mqtt.kalive',
    'mqtt.retain', 'tcp.time_delta', 'mqtt.conflag.cleansess',
    'mqtt.proto_len', 'mqtt.ver'
]

def convert_hex_columns(df, columns):
    for col in columns:
        df[col] = df[col].apply(
            lambda x: int(x, 16) if isinstance(x, str) and x.startswith('0x') else 0
        )
    return df

hex_columns = ['tcp.flags', 'mqtt.hdrflags']
def main():
    p = argparse.ArgumentParser(
        description="Run TFLite model on Raspberry Pi and optionally save confusion matrix"
    )
    p.add_argument("-m", "--model-tflite", required=True,
                   help="Path to the .tflite model file")
    p.add_argument("-s", "--scaler-pkl", required=True,
                   help="Path to the StandardScaler .pkl used at training")
    p.add_argument("-l", "--label-encoder-pkl", required=True,
                   help="Path to the LabelEncoder .pkl used at training")
    p.add_argument("-i", "--input", required=True,
                   help="Input CSV with FEATURES columns (and optional 'target')")
    p.add_argument("-o", "--output", default=None,
                   help="(Optional) CSV path to save predictions")
    p.add_argument("-c", "--cm", default=None,
                   help="(Optional) PNG path to save confusion matrix")
    args = p.parse_args()

    # Load scaler and label encoder
    scaler = joblib.load(args.scaler_pkl)
    le = joblib.load(args.label_encoder_pkl)

    # Read input data
    df = pd.read_csv(args.input)
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    df = convert_hex_columns(df, hex_columns)
    df = df.replace({'True': 1, 'False': 0})
    # Extract and scale features
    X = df[FEATURES].to_numpy().astype(np.float32)
    X_scaled = scaler.transform(X)

    # Load TFLite interpreter
    if not os.path.isfile(args.model_tflite):
        raise FileNotFoundError(f"Model not found: {args.model_tflite}")
    interpreter = Interpreter(model_path=args.model_tflite)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Run inference sample-by-sample
    preds_idx = []
    for sample in X_scaled:
        inp = np.expand_dims(sample, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details['index'], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details['index'])
        preds_idx.append(int(np.argmax(out[0])))
    preds_idx = np.array(preds_idx)

    # Decode to original labels
    preds_labels = le.inverse_transform(preds_idx)
    df['prediction'] = preds_labels

    # Save predictions CSV
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")

    # If ground truth present and -c given, compute & save confusion matrix
    if 'target' in df.columns and args.cm:
        y_true = df['target']
        labels = list(le.classes_)
        cm = confusion_matrix(y_true, preds_labels, labels=labels)
        print("=== Confusion Matrix ===")
        print(pd.DataFrame(cm, index=labels, columns=labels))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots()
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(args.cm)
        plt.close(fig)
        print(f"Confusion matrix saved to {args.cm}")

if __name__ == "__main__":
    main()
