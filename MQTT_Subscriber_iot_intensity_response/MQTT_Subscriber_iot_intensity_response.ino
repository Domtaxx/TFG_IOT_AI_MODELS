#include <WiFi.h>
#include <PubSubClient.h>

// WiFi & MQTT Configuration
const char* ssid = "WQ Router";
const char* password = "UliMar17";
const char* mqtt_server = "raspberrypi.local";  // e.g., "192.168.1.100"

WiFiClient espClient;
PubSubClient client(espClient);

// Topics
const char* topic_sub = "iot/#";           // Subscribe to this topic
const char* topic_pub = "iot/intensity-response";  // Publish response here

// 🔹 Function to Connect to WiFi
void setup_wifi() {
    Serial.print("Connecting to WiFi...");
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi Connected!");
}

// 🔹 Function to Reconnect to MQTT if Disconnected
void reconnect_mqtt() {
    while (!client.connected()) {
        Serial.print("Connecting to MQTT...");
        if (client.connect("ESP32_Subscriber")) {  // Client ID must be unique
            Serial.println("Connected!");
            client.subscribe(topic_sub);  // Subscribe to intensity topic
        } else {
          Serial.println(" Retrying in 2 seconds...");
          delay(2000);
        }
    }
}

// 🔹 Callback Function - Triggered when a message is received
void callback(char* topic, byte* payload, unsigned int length) {
    Serial.print("Message received on topic: ");
    Serial.println(topic);

    // Convert payload to string
    String receivedMessage = "Message Received";
    Serial.println("Message Received");
    delay(1000); 
    //Re-Publish to `iot/intensity-response`
    client.publish(topic_pub, receivedMessage.c_str());
    Serial.println("Re-Published to " + String(topic_pub));
}

// 🔹 Setup Function
void setup() {
    Serial.begin(115200);
    setup_wifi();
    client.setServer(mqtt_server, 1883);
    client.setKeepAlive(60);  // Tiempo Keep Alive en segundos
    client.setCallback(callback);  // Attach callback function
}

// 🔹 Main Loop
void loop() {
    if (!client.connected()) {
        reconnect_mqtt();
    }
    client.loop();  // Keep MQTT connection alive
}
