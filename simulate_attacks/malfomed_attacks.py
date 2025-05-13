import socket
import random
import time
import struct
import csv
import os

# 🔧 Settings
BROKER_HOST = "raspberrypi.local"
BROKER_PORT = 1883
PACKET_COUNT = 5000
DELAY_BETWEEN_PACKETS = 0.1
CSV_LOG_FILE = "malformed_attacks_log.csv"

# 🧾 CSV columns + target
CSV_COLUMNS = [
    'tcp.flags','tcp.time_delta','tcp.len','mqtt.conack.flags','mqtt.conack.flags.reserved','mqtt.conack.flags.sp',
    'mqtt.conack.val','mqtt.conflag.cleansess','mqtt.conflag.passwd','mqtt.conflag.qos','mqtt.conflag.reserved',
    'mqtt.conflag.retain','mqtt.conflag.uname','mqtt.conflag.willflag','mqtt.conflags','mqtt.dupflag','mqtt.hdrflags',
    'mqtt.kalive','mqtt.len','mqtt.topic','mqtt.msg','mqtt.msgid','mqtt.msgtype','mqtt.proto_len','mqtt.protoname',
    'mqtt.qos','mqtt.retain','mqtt.sub.qos','mqtt.suback.qos','mqtt.ver','mqtt.willmsg','mqtt.willmsg_len',
    'mqtt.willtopic','mqtt.willtopic_len','target'
]

# 🧾 Initialize CSV
# 🧾 Initialize CSV with header (even if empty file already exists)
if not os.path.isfile(CSV_LOG_FILE) or os.stat(CSV_LOG_FILE).st_size == 0:
    with open(CSV_LOG_FILE, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

# 🧠 Hybrid malformed PUBLISH (valid topic, bad payload)
def generate_hybrid_publish_packet():
    topic = b"malformed/test"
    topic_len = len(topic)

    # Corrupted binary payload (not valid UTF-8)
    payload = bytes([0xFF, 0xFE, 0x00, 0x80, 0xAB, 0xCD])
    remaining_length = 2 + topic_len + len(payload)

    packet = bytearray()
    packet.append(0x30)  # PUBLISH
    packet.append(remaining_length)
    packet.extend(struct.pack("!H", topic_len))  # topic length
    packet.extend(topic)
    packet.extend(payload)
    return packet, topic.decode(), payload.hex()

# 🧾 Log attack to CSV
def log_attack(packet_len, topic, payload_hex):
    row = {
        'tcp.flags': "0x00",
        'tcp.time_delta': round(random.uniform(0.00001, 0.05), 8),
        'tcp.len': packet_len,
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
        'mqtt.kalive': 0,
        'mqtt.len': packet_len,
        'mqtt.topic': topic,
        'mqtt.msg': payload_hex,
        'mqtt.msgid': random.randint(1, 10000),
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
        'target': 'malformed'
    }

    with open(CSV_LOG_FILE, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writerow(row)

# 🚀 Main attack loop
def run_attack():
    print(f"💥 Sending hybrid malformed MQTT PUBLISH packets to {BROKER_HOST}:{BROKER_PORT}")
    for i in range(PACKET_COUNT):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((BROKER_HOST, BROKER_PORT))

            packet, topic_str, payload_hex = generate_hybrid_publish_packet()
            sock.sendall(packet)
            log_attack(len(packet), topic_str, payload_hex)

            print(f"✅ Packet {i+1} sent → Topic: {topic_str}, Payload: {payload_hex}")
            sock.close()
            time.sleep(DELAY_BETWEEN_PACKETS)
        except Exception as e:
            print(f"❌ Packet {i+1} failed: {e}")

if __name__ == "__main__":
    run_attack()
