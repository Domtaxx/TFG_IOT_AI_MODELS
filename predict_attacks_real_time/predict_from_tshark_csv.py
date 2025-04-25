import subprocess
import pandas as pd
import joblib
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
from time import time
from io import StringIO
import json
import re

# Cargar modelo y scaler
model = joblib.load("./random_forest_model.pkl")
scaler = joblib.load("./scaler.pkl")
label_encoder = joblib.load("./label_encoder.pkl")
# Columnas exactas esperadas por el modelo
expected_columns = [
    "tcp.time_delta",
    "tcp.len",
    "mqtt.conack.flags.reserved",
    "mqtt.conack.flags.sp",
    "mqtt.conack.val",
    "mqtt.conflag.cleansess",
    "mqtt.conflag.passwd",
    "mqtt.conflag.qos",
    "mqtt.conflag.reserved",
    "mqtt.conflag.retain",
    "mqtt.conflag.uname",
    "mqtt.conflag.willflag",
    "mqtt.dupflag",
    "mqtt.kalive",
    "mqtt.len",
    "mqtt.msgid",
    "mqtt.msgtype",
    "mqtt.proto_len",
    "mqtt.qos",
    "mqtt.retain",
    "mqtt.sub.qos",
    "mqtt.suback.qos",
    "mqtt.willmsg_len",
    "mqtt.willtopic_len"
]

# MQTT para alertas
ALERT_TOPIC = "iot/anomalies"
mqtt_client = mqtt.Client()
mqtt_client.connect("localhost", 1883, 60)
mqtt_client.loop_start()

# Comando tshark como subprocess (modificá la interfaz si hace falta)
tshark_fields = [
    "tcp.flags",
    "tcp.time_delta",
    "tcp.len",
    "mqtt.conack.flags",
    "mqtt.conack.flags.reserved",
    "mqtt.conack.flags.sp",
    "mqtt.conack.val",
    "mqtt.conflag.cleansess",
    "mqtt.conflag.passwd",
    "mqtt.conflag.qos",
    "mqtt.conflag.reserved",
    "mqtt.conflag.retain",
    "mqtt.conflag.uname",
    "mqtt.conflag.willflag",
    "mqtt.conflags",
    "mqtt.dupflag",
    "mqtt.hdrflags",
    "mqtt.kalive",
    "mqtt.len",
    "mqtt.topic",
    "mqtt.msg",
    "mqtt.msgid",
    "mqtt.msgtype",
    "mqtt.proto_len",
    "mqtt.protoname",
    "mqtt.qos",
    "mqtt.retain",
    "mqtt.sub.qos",
    "mqtt.suback.qos",
    "mqtt.ver",
    "mqtt.willmsg",
    "mqtt.willmsg_len",
    "mqtt.willtopic",
    "mqtt.willtopic_len"
]

# Comando tshark limpio y dinámico
tshark_cmd = [
    "sudo",
    "tshark",
    "-l",
    "-i", "wlan0",                         # Cambiar por tu interfaz de red real
    "-f", "tcp port 1883",               # Filtro BPF para tráfico MQTT
    "-Y", 'mqtt.msgtype==3',                        # Filtro de protocolo para mqtt
    "-T", "fields"
]

# Agregar todos los campos con '-e'
for field in tshark_fields:
    tshark_cmd.extend(["-e", field])

# Formato del output
tshark_cmd += [
    "-E", "header=y",
    "-E", "separator=,", 
    "-E", "quote=d",
    "-E", "occurrence=f"
]

def safe_str(val):
    return str(val) if pd.notna(val) else "none"

print("Iniciando captura en tiempo real con tshark...")
process = subprocess.Popen(tshark_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
MAX_PROCESSED = 10
processed_messages = set()
count = 0
simulation_data = {
    "legitimate": {"legitimate":0, "dos":0, "flood":0, "malformed": 0, "bruteforce": 0},
    "dos": {"legitimate":0, "dos":0, "flood":0, "malformed": 0, "bruteforce": 0},
    "flood": {"legitimate":0, "dos":0, "flood":0, "malformed": 0, "bruteforce": 0},
    "malformed": {"legitimate":0, "dos":0, "flood":0, "malformed": 0, "bruteforce": 0}
}
for line in process.stdout:
    try:
        if count == 50:
            with open("data.json", 'w') as file:
                json.dump(simulation_data, file, indent=4)
            process.terminate()
            exit()
        line = line.strip()
       # print(line)
        if not line:
            continue
        # Convertimos la línea a dataframe temporal
        df = pd.read_csv(StringIO(line), names=tshark_fields)
        df = df.fillna(0.0)
        # Asegurar orden de columnas
        categorical_columns = ['tcp.flags', 'mqtt.conack.flags', 'mqtt.conflags', 'mqtt.hdrflags', 
                    'mqtt.msg', 'mqtt.protoname', 'mqtt.ver', 'mqtt.willmsg', 'mqtt.willtopic', "mqtt.topic"]
        topic = safe_str(df["mqtt.topic"][0]) if "mqtt.topic" in df else "none"
        try:
            msg = bytes.fromhex(df["mqtt.msg"][0]).decode('ascii')
        except Exception:
        # If it fails, return original
            msg = df["mqtt.msg"][0]
        msg_id = safe_str(df["mqtt.msgid"][0]) if "mqtt.msgid" in df else "none"
        msg_key = str(topic)+str(msg_id)+str(msg)
        if len(processed_messages) > MAX_PROCESSED:
            processed_messages.clear()	
        if msg_key in processed_messages:
            continue
        else:
            df = df.drop(columns=categorical_columns, errors='ignore')
            # Convertir a float de forma segura
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # Preprocesamiento y predicción
            rescaled_data = scaler.transform(df)
            pred = model.predict(rescaled_data)
            pred_labels = label_encoder.inverse_transform(pred)
            df["prediction"] = pred
            df["label"] = pred_labels
            topic_split_by_slash = topic.split("/")
            if pred_labels[0] != "legitimate":
                print(f"Anomalia {pred_labels[0]} detectada en topico:{topic} → msg:{msg}")
                if topic_split_by_slash[1] == "attack":
                    attack_type = topic_split_by_slash[2]
                    try:
                        simulation_data[attack_type][pred_labels[0]] += 1
                    except:
                        simulation_data[attack_type]["malformed"] += 1
                else:
                    simulation_data["legitimate"][pred_labels[0]] += 1
            else:
                if topic_split_by_slash[1] == "attack":
                    attack_type = topic_split_by_slash[2]
                    try:
                        simulation_data[attack_type][pred_labels[0]] += 1
                    except:
                        simulation_data[attack_type]["malformed"] += 1
                else:
                    simulation_data["legitimate"][pred_labels[0]] += 1
                print(f"Normal en topico {topic}")
            processed_messages.add(msg_key)
            count += 1
            print(f"count when messages start:{count}")
    except Exception as e:
        print(f"[Error procesando línea]: {e} → {line}")
#    print(process.stderr)   
