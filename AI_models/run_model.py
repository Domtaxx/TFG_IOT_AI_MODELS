#!/usr/bin/env python3
import os
import pandas as pd
import joblib
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# same FEATURES list you used for training
FEATURES = [
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
    'mqtt.conack.flags.sp'
]
def convert_hex_columns(df, columns):
    for col in columns:
        df[col] = df[col].apply(
            lambda x: int(x, 16) if isinstance(x, str) and x.startswith('0x') else 0
        )
    return df

hex_columns = ['tcp.flags', 'mqtt.hdrflags']
def main():
    p = argparse.ArgumentParser(description="Run RF .pkl and optional Confusion Matrix")
    p.add_argument("-m","--model-pkl", required=True,
                   help="Path to joblib .pkl of your RandomForest")
    p.add_argument("-i","--input", required=True,
                   help="Input CSV with FEATURES (and optional 'target')")
    p.add_argument("-o","--output", default=None,
                   help="(Optional) output CSV path to save predictions")
    p.add_argument("-c","--cm", default=None,
                   help="(Optional) PNG path to save confusion matrix")
    args = p.parse_args()

    # load model
    if not os.path.isfile(args.model_pkl):
        raise FileNotFoundError(f"No existe {args.model_pkl}")
    model = joblib.load(args.model_pkl)

    # read data
    df = pd.read_csv(args.input)
    if not set(FEATURES).issubset(df.columns):
        missing = set(FEATURES) - set(df.columns)
        raise ValueError(f"Faltan columnas de características: {missing}")
    
    df = convert_hex_columns(df, hex_columns)
    df = df.replace({'True': 1, 'False': 0})
    X = df[FEATURES].to_numpy()


    # predict
    preds = model.predict(X)
    df["prediction"] = preds

    # save predictions
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Predicciones guardadas en {args.output}")

    # confusion matrix
    if "target" in df.columns and args.cm:
        y_true = df["target"]
        y_pred = df["prediction"]
        labels = sorted(df["target"].unique())
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # console print
        print("=== Matriz de Confusión ===")
        print(pd.DataFrame(cm, index=labels, columns=labels))

        # plot + save PNG with white→blue cmap
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots()
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title("Matriz de Confusión")
        fig.tight_layout()
        fig.savefig(args.cm)
        plt.close(fig)
        print(f"Matriz de confusión guardada en: {args.cm}")

if __name__ == "__main__":
    main()
