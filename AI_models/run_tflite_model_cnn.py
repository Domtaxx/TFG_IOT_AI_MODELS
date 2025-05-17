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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

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

def main():
    p = argparse.ArgumentParser(
        description="Run TFLite CNN with per-sample performance monitoring"
    )
    p.add_argument("-m", "--model-tflite", required=True, help="Path to the .tflite model")
    p.add_argument("-s", "--scaler-pkl",   required=True, help="Path to the scaler .pkl")
    p.add_argument("-l", "--label-encoder-pkl", required=True, help="Path to the label encoder .pkl")
    p.add_argument("-i", "--input",        required=True, help="Input CSV with FEATURES and optional 'target'")
    p.add_argument("-o", "--output",       default=None,   help="(Optional) CSV to save predictions")
    p.add_argument("-c", "--cm",           default=None,   help="(Optional) PNG to save confusion matrix")
    args = p.parse_args()

    # Prepare process for CPU/RAM metrics
    proc = psutil.Process(os.getpid())
    metrics = []

    # Load preprocessing artifacts
    scaler = joblib.load(args.scaler_pkl)
    le     = joblib.load(args.label_encoder_pkl)

    # Read and preprocess input
    df = pd.read_csv(args.input)
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    df = convert_hex_columns(df, ['tcp.flags', 'mqtt.hdrflags'])
    df = df.replace({'True':1,'False':0})

    # Extract & scale features
    X = df[FEATURES].to_numpy().astype(np.float32)
    X_scaled = scaler.transform(X)

    # Load TFLite model
    interpreter = Interpreter(model_path=args.model_tflite)
    interpreter.allocate_tensors()
    inp_det  = interpreter.get_input_details()[0]
    out_det  = interpreter.get_output_details()[0]
    _, timesteps, channels = inp_det['shape']

    # Inference with per-sample metrics
    start_all = time.perf_counter()
    for idx, sample in enumerate(X_scaled):
        wall_start = time.perf_counter()
        cpu_start  = time.process_time()
        mem_start  = proc.memory_info().rss

        inp = sample.reshape((1, timesteps, channels))
        interpreter.set_tensor(inp_det['index'], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(out_det['index'])
        # collect prediction
        pred_idx = int(np.argmax(out[0]))

        wall_end = time.perf_counter()
        cpu_end  = time.process_time()
        mem_end  = proc.memory_info().rss

        dt_wall = wall_end - wall_start
        dt_cpu  = cpu_end  - cpu_start
        cpu_pct = (dt_cpu / dt_wall * 100) if dt_wall > 0 else 0.0
        ram_mb  = (mem_end - mem_start) / (1024**2)

        metrics.append({
            'sample': idx,
            'wall_time_s': dt_wall,
            'cpu_time_s': dt_cpu,
            'cpu_percent': cpu_pct,
            'ram_change_mb': ram_mb,
            'pred_idx': pred_idx
        })

    total_wall = time.perf_counter() - start_all

    # Build predictions column
    preds_idx = [m['pred_idx'] for m in metrics]
    preds_labels = le.inverse_transform(preds_idx)
    df['prediction'] = preds_labels

    # Save per-sample metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_csv = os.path.splitext(args.input)[0] + '_metrics.csv'
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved per-sample metrics to {metrics_csv}")

    # Compute and print averages
    avg = metrics_df[['wall_time_s','cpu_time_s','cpu_percent','ram_change_mb']].mean()
    print("\n--- Average Inference Metrics ---")
    print(f"Avg wall time (s): {avg.wall_time_s:.6f}")
    print(f"Avg CPU time  (s): {avg.cpu_time_s:.6f}")
    print(f"Avg CPU %       : {avg.cpu_percent:.2f}%")
    print(f"Avg RAM change (MB): {avg.ram_change_mb:.4f}")
    print(f"Total wall time (s): {total_wall:.4f}")

    # Save predictions if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")

    

    # Confusion matrix if ground truth present
    if 'target' in df.columns and args.cm:
        y_true = df['target']
        y_pred = df['prediction']
        print("=== Classification Report ===")
        print(classification_report(y_true, y_pred, target_names=le.classes_))

        cm = confusion_matrix(y_true, preds_labels, labels=list(le.classes_))
        print("\n=== Confusion Matrix ===")
        print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

        disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
        fig, ax = plt.subplots()
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(args.cm)
        plt.close(fig)
        print(f"Confusion matrix saved to {args.cm}")

if __name__ == "__main__":
    main()

