#!/usr/bin/env python3
import os
import time
import pandas as pd
import numpy as np
import psutil
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

FEATURES = [
    'mqtt.msgid', 'mqtt.qos', 'mqtt.len', 'tcp.len', 'tcp.flags',
    'mqtt.hdrflags', 'mqtt.msgtype', 'mqtt.dupflag', 'mqtt.kalive',
    'mqtt.retain', 'tcp.time_delta', 'mqtt.conflag.cleansess',
    'mqtt.proto_len', 'mqtt.conack.val', 'mqtt.conflag.passwd', 
    'mqtt.conflag.uname', 'mqtt.ver', 'mqtt.conflags','mqtt.msg', 
    'mqtt.protoname'
]
def preprocess_data(df):
    df = df.copy()
    # Don't convert 'target' to category
    cat_columns = df.drop(columns=['target']).select_dtypes(include=['object', 'category']).columns
    df[cat_columns] = df[cat_columns].astype('category')
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df

def main():
    model_type = 'cnn'
    training_size = 1
    # === Set paths here instead of argparse ===
    MODEL_TFLITE_PATH = f"{model_type}_neural_network/{int(training_size*100)}/{model_type}_model.tflite"
    LABEL_ENCODER_PKL_PATH = f"{model_type}_neural_network/{int(training_size*100)}/{model_type}_label_encoder.pkl"
    INPUT_CSV = f"results_merged.csv"
    OUTPUT_CSV = f"{model_type}_neural_network/{int(training_size*100)}/output_predictions.csv"
    CONFUSION_MATRIX_PATH = f"{model_type}_neural_network/{int(training_size*100)}/{model_type}_{int(training_size*100)}_confusion_matrix_results_3.png"

    proc = psutil.Process(os.getpid())
    metrics = []

    # Load scaler & label encoder
    le = joblib.load(LABEL_ENCODER_PKL_PATH)

    # Read and preprocess CSV
    df = pd.read_csv(INPUT_CSV)
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    df = preprocess_data(df)

    X = df[FEATURES].to_numpy().astype(np.float32)

    # Load TFLite model
    if not os.path.isfile(MODEL_TFLITE_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_TFLITE_PATH}")
    interpreter = Interpreter(model_path=MODEL_TFLITE_PATH)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    inp_shape = inp_det['shape']
    if len(inp_shape) == 3:
        _, timesteps, channels = inp_shape
        reshape_fn = lambda sample: sample.reshape((1, timesteps, channels))
    elif len(inp_shape) == 2:
        _, features = inp_shape
        reshape_fn = lambda sample: sample.reshape((1, features))
    else:
        raise ValueError(f"Unsupported input shape: {inp_shape}")
    
    
    total_start = time.perf_counter()
    cpu_start_global = proc.cpu_times().user + proc.cpu_times().system
    mem_start_global = proc.memory_info().rss
    for idx, sample in enumerate(X):
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        mem_start = proc.memory_info().rss

        inp_tensor = reshape_fn(sample).astype(np.float32)
        interpreter.set_tensor(inp_det['index'], inp_tensor)
        interpreter.invoke()
        out = interpreter.get_tensor(out_det['index'])
        pred_idx = int(np.argmax(out[0]))

        wall_end = time.perf_counter()
        cpu_end = time.process_time()
        mem_end = proc.memory_info().rss

        dt_wall = wall_end - wall_start
        dt_cpu = cpu_end - cpu_start
        cpu_pct = (dt_cpu / dt_wall * 100) if dt_wall > 0 else 0.0
        ram_mb = (mem_end - mem_start) / (1024**2)

        metrics.append({
            'sample': idx,
            'wall_time_s': dt_wall,
            'cpu_time_s': dt_cpu,
            'cpu_percent': cpu_pct,
            'ram_change_mb': ram_mb,
            'pred_idx': pred_idx
        })
    total_wall = time.perf_counter() - total_start

    preds_idx = [m['pred_idx'] for m in metrics]
    preds_labels = le.inverse_transform(preds_idx)
    df['prediction'] = preds_labels

    # Save per-sample metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_csv = os.path.splitext(INPUT_CSV)[0] + '_metrics.csv'
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved metrics to {metrics_csv}")

    # Print average metrics
    n = len(X)
    cpu_end = proc.cpu_times().user + proc.cpu_times().system
    mem_end = proc.memory_info().rss

    total_time = total_wall  # already measured
    cpu_time = cpu_end - cpu_start_global  # compute total CPU time
    avg_time_ms = (total_time / n) * 1000
    cpu_percent = (cpu_time / total_time) * 100 if total_time > 0 else 0
    ram_peak_mb = (mem_end - mem_start_global) / (1024**2)
    avg_ram_mb = ram_peak_mb / n

    print("\n=== Métricas de Inference ===")
    print(f"Total muestras         : {n}")
    print(f"Tiempo total           : {total_time:.4f} s")
    print(f"Tiempo medio/muestra   : {avg_time_ms:.4f} ms")
    print(f"CPU time total         : {cpu_time:.4f} s")
    print(f"% CPU promedio         : {cpu_percent:.2f} %")
    print(f"RAM Peak               : {ram_peak_mb:.6f} MB")
    print(f"RAM promedio/muestra   : {avg_ram_mb:.6f} MB\n")

    #if OUTPUT_CSV:
    #    df.to_csv(OUTPUT_CSV, index=False)
    #    print(f"Predictions saved to {OUTPUT_CSV}")

    if 'target' in df.columns and CONFUSION_MATRIX_PATH:
        y_true = df['target']
        print("\n=== Classification Report ===")
        print(classification_report(y_true, df['prediction'], target_names=le.classes_))

        cm = confusion_matrix(y_true, df['prediction'], labels=le.classes_)
        print("\n=== Confusion Matrix ===")
        print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        fig, ax = plt.subplots()
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(CONFUSION_MATRIX_PATH)
        plt.close(fig)
        print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")

if __name__ == "__main__":
    main()
