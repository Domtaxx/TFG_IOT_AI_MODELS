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

expected_columns = [
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
    training_size = 0.25
    MODEL_PATH = f"random_forest/{int(training_size*100)}/random_forest.pkl"
    ENCODER_PATH = f"random_forest/{int(training_size*100)}/label_encoder.pkl"
    INPUT_CSV = f"results_merged.csv"
    OUTPUT_CSV = f"random_forest/{int(training_size*100)}/predictions.csv"
    CONFUSION_MATRIX_PATH = f"random_forest/{int(training_size*100)}/rf_{int(training_size*100)}_test_confusion_matrix_3.png"

    # Cargar modelo y preprocesadores
    model  = joblib.load(MODEL_PATH)
    le     = joblib.load(ENCODER_PATH)

    # Leer datos
    df = pd.read_csv(INPUT_CSV, header=0)
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas de características: {missing}")

    # Preprocesar
    df = preprocess_data(df)
    
    # Extraer y escalar features
    X = df[expected_columns]#.drop('target', axis=1).values#

    # Preparar medición de CPU y RAM
    proc = psutil.Process(os.getpid())
    cpu_start = proc.cpu_times().user + proc.cpu_times().system
    t_start   = time.perf_counter()
    mem_start = proc.memory_info().rss

    # Inferencia en lote
    preds_idx = model.predict(X)

    t_end   = time.perf_counter()
    cpu_end = proc.cpu_times().user + proc.cpu_times().system
    mem_end = proc.memory_info().rss

    # Cálculo de métricas
    n = len(preds_idx)
    total_time    = t_end - t_start
    cpu_time      = cpu_end - cpu_start
    avg_time_ms   = (total_time / n) * 1000
    cpu_percent   = (cpu_time / total_time) * 100 if total_time > 0 else 0
    avg_ram_mb    = ((mem_end - mem_start) / (1024**2)) / n

    print("\n=== Métricas de Inference ===")
    print(f"Total muestras         : {n}")
    print(f"Tiempo total           : {total_time:.4f} s")
    print(f"Tiempo medio/muestra   : {avg_time_ms:.4f} ms")
    print(f"CPU time total         : {cpu_time:.4f} s")
    print(f"% CPU promedio         : {cpu_percent:.2f} %")
    print(f"RAM Peak               : {((mem_end - mem_start) / (1024**2)):.6f} MB")
    print(f"RAM promedio/muestra   : {avg_ram_mb:.6f} MB\n")

    # Decodificar predicciones
    preds_idx = preds_idx.astype(np.int32)
    preds = le.inverse_transform(preds_idx)
    df['prediction'] = preds

    # Guardar predicciones
    #if OUTPUT_CSV:
    #    df.to_csv(OUTPUT_CSV, index=False)
    #    print(f"Predicciones guardadas en: {OUTPUT_CSV}")

    # Reporte y matriz de confusión
    if 'target' in df.columns and CONFUSION_MATRIX_PATH:
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
        fig.savefig(CONFUSION_MATRIX_PATH)
        plt.close(fig)
        print(f"Matriz de confusión guardada en: {CONFUSION_MATRIX_PATH}")

if __name__ == "__main__":
    main()
