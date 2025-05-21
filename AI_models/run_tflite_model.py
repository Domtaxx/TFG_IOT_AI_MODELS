#!/usr/bin/env python3
import os
import time
import argparse
import pandas as pd
import numpy as np
import psutil
from tflite_runtime.interpreter import Interpreter
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# List of feature column names, same order used at training
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
        description="Run any TFLite model (CNN or Dense NN) with per-sample performance monitoring"
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

    # Setup psutil for CPU/RAM monitoring
    proc = psutil.Process(os.getpid())
    metrics = []

    # Load scaler & label encoder
    scaler = joblib.load(args.scaler_pkl)
    le     = joblib.load(args.label_encoder_pkl)

    # Read & preprocess input CSV
    df = pd.read_csv(args.input)
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    df = convert_hex_columns(df, hex_columns)
    df = df.replace({'True':1,'False':0})

    # Extract & scale features
    X = df[FEATURES].to_numpy().astype(np.float32)
    X_scaled = scaler.transform(X)

    # Load TFLite model
    if not os.path.isfile(args.model_tflite):
        raise FileNotFoundError(f"Model not found: {args.model_tflite}")
    interpreter = Interpreter(model_path=args.model_tflite)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    # Inspect input shape
    inp_shape = inp_det['shape']
    if len(inp_shape) == 3:
        # CNN: [batch, timesteps, channels]
        _, timesteps, channels = inp_shape
        reshape_fn = lambda sample: sample.reshape((1, timesteps, channels))
    elif len(inp_shape) == 2:
        # Dense NN: [batch, features]
        _, features = inp_shape
        reshape_fn = lambda sample: sample.reshape((1, features))
    else:
        raise ValueError(f"Unsupported input shape: {inp_shape}")

    # Inference with per-sample metrics
    total_start = time.perf_counter()
    for idx, sample in enumerate(X_scaled):
        wall_start = time.perf_counter()
        cpu_start  = time.process_time()
        mem_start  = proc.memory_info().rss

        inp_tensor = reshape_fn(sample).astype(np.float32)
        interpreter.set_tensor(inp_det['index'], inp_tensor)
        interpreter.invoke()
        out = interpreter.get_tensor(out_det['index'])
        pred_idx = int(np.argmax(out[0]))

        wall_end = time.perf_counter()
        cpu_end  = time.process_time()
        mem_end  = proc.memory_info().rss

        dt_wall = wall_end - wall_start
        dt_cpu  = cpu_end - cpu_start
        cpu_pct = (dt_cpu / dt_wall * 100) if dt_wall > 0 else 0.0
        ram_mb  = (mem_end - mem_start) / (1024**2)

        metrics.append({
            'sample': idx,
            'wall_time_s': dt_wall,
            'cpu_time_s':   dt_cpu,
            'cpu_percent':  cpu_pct,
            'ram_change_mb': ram_mb,
            'pred_idx':     pred_idx
        })
    total_wall = time.perf_counter() - total_start

    # Build predictions
    preds_idx    = [m['pred_idx'] for m in metrics]
    preds_labels = le.inverse_transform(preds_idx)
    df['prediction'] = preds_labels

    # Save metrics CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_csv = os.path.splitext(args.input)[0] + '_metrics.csv'
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved per-sample metrics to {metrics_csv}")

    # Print average metrics
    avg = metrics_df[['wall_time_s','cpu_time_s','cpu_percent','ram_change_mb']].mean()
    print("\n--- Average Inference Metrics ---")
    print(f"Avg wall time (s)     : {avg.wall_time_s:.6f}")
    print(f"Avg CPU time  (s)     : {avg.cpu_time_s:.6f}")
    print(f"Avg CPU %             : {avg.cpu_percent:.2f}%")
    print(f"Avg RAM change (MB)   : {avg.ram_change_mb:.4f}")
    print(f"Total inference wall  : {total_wall:.4f} s\n")

    # Save predictions CSV
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")

    # Classification report & confusion matrix
    if 'target' in df.columns and args.cm:
        y_true = df['target']
        print("=== Classification Report ===")
        print(classification_report(y_true, df['prediction'], target_names=le.classes_))

        cm = confusion_matrix(y_true, df['prediction'], labels=list(le.classes_))
        print("\n=== Confusion Matrix ===")
        print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        fig, ax = plt.subplots()
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(args.cm)
        plt.close(fig)
        print(f"Confusion matrix saved to {args.cm}")

if __name__ == "__main__":
    main()
