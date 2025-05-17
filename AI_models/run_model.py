#!/usr/bin/env python3
import os
import argparse
import time
import psutil
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Lista de columnas de entrada en el mismo orden que en el entrenamiento
FEATURES = [
    'mqtt.msgid', 'mqtt.qos', 'mqtt.len', 'tcp.len', 'tcp.flags',
    'mqtt.hdrflags', 'mqtt.msgtype', 'mqtt.dupflag', 'mqtt.kalive',
    'mqtt.retain', 'tcp.time_delta', 'mqtt.conflag.cleansess',
    'mqtt.proto_len', 'mqtt.ver'
]
HEX_COLUMNS = ['tcp.flags', 'mqtt.hdrflags']

def convert_hex_columns(df, cols):
    for col in cols:
        df[col] = df[col].apply(
            lambda x: int(x, 16) if isinstance(x, str) and x.startswith('0x') else int(x)
        )
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Run RandomForest .pkl con métricas de tiempo, CPU y RAM"
    )
    parser.add_argument("-m","--model-pkl",       required=True, help="Path to RandomForest .pkl (joblib)")
    parser.add_argument("-s","--scaler-pkl",      required=True, help="Path to StandardScaler .pkl")
    parser.add_argument("-l","--label-encoder-pkl", required=True, help="Path to LabelEncoder .pkl")
    parser.add_argument("-i","--input",           required=True, help="Input CSV with FEATURES (and optional 'target')")
    parser.add_argument("-o","--output",          default=None, help="(Optional) CSV path to save predictions")
    parser.add_argument("-c","--cm",              default=None, help="(Optional) PNG path to save confusion matrix")
    args = parser.parse_args()

    # Cargar modelo y preprocesadores
    model  = joblib.load(args.model_pkl)
    scaler = joblib.load(args.scaler_pkl)
    le     = joblib.load(args.label_encoder_pkl)

    # Leer datos
    df = pd.read_csv(args.input)
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas de características: {missing}")

    # Preprocesar
    df = convert_hex_columns(df, HEX_COLUMNS)
    df = df.replace({'True': 1, 'False': 0})

    # Extraer y escalar features
    X = df[FEATURES].to_numpy().astype(np.float32)
    X_scaled = scaler.transform(X)

    # Preparar medición de CPU y RAM
    proc = psutil.Process(os.getpid())
    cpu_start = proc.cpu_times().user + proc.cpu_times().system
    t_start   = time.perf_counter()
    mem_start = proc.memory_info().rss

    # Inferencia en lote
    preds_idx = model.predict(X_scaled)

    t_end   = time.perf_counter()
    cpu_end = proc.cpu_times().user + proc.cpu_times().system
    mem_end = proc.memory_info().rss

    # Cálculo de métricas
    n = len(preds_idx)
    total_time    = t_end - t_start
    cpu_time      = cpu_end - cpu_start
    avg_time_ms   = (total_time / n) * 1000
    cpu_percent   = (cpu_time / total_time) * 100 if total_time > 0 else 0
    avg_ram_mb    = (mem_end / (1024**2)) / n

    print("\n=== Métricas de Inference ===")
    print(f"Total muestras         : {n}")
    print(f"Tiempo total           : {total_time:.4f} s")
    print(f"Tiempo medio/muestra   : {avg_time_ms:.4f} ms")
    print(f"CPU time total         : {cpu_time:.4f} s")
    print(f"% CPU promedio         : {cpu_percent:.2f} %")
    print(f"RAM Peak               : {(mem_end / (1024**2)):.6f} MB")
    print(f"RAM promedio/muestra   : {avg_ram_mb:.6f} MB\n")

    # Decodificar predicciones
    preds_idx = preds_idx.astype(np.int32)
    preds = le.inverse_transform(preds_idx)
    df['prediction'] = preds

    # Guardar predicciones
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Predicciones guardadas en: {args.output}")

    # Reporte y matriz de confusión
    if 'target' in df.columns and args.cm:
        y_true = df['target']
        y_pred = df['prediction']

        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")

        print("=== Classification Report ===")
        print(classification_report(y_true, y_pred, target_names=le.classes_))

        labels = list(le.classes_)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print("=== Matriz de Confusión ===")
        print(pd.DataFrame(cm, index=labels, columns=labels))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots()
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title("Matriz de Confusión")
        fig.tight_layout()
        fig.savefig(args.cm)
        plt.close(fig)
        print(f"Matriz de confusión guardada en: {args.cm}")

if __name__ == "__main__":
    main()
