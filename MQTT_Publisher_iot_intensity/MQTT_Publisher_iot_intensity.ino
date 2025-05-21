#include <WiFi.h>
#include <PubSubClient.h>

const char* ssid = "WQ Router";
const char* password = "UliMar17";
const char* mqtt_server = "raspberrypi.local";  // e.g., "192.168.1.100"
const char* mqtt_username = "domtaxx";
const char* mqtt_password = "TFG2025";
WiFiClient espClient;
PubSubClient client(espClient);

void setup_wifi() {
  delay(10);
  Serial.println("\nConnecting to WiFi...");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\nWiFi connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect("ESP32Client_iot_intensity", mqtt_username, mqtt_password)) {
      Serial.println("Connected!");
    } else {
      Serial.print("Failed, rc=");
      Serial.print(client.state());
      Serial.println(" Retrying in 2 seconds...");
      delay(2000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setKeepAlive(60);  // Tiempo Keep Alive en segundos
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }

  int randomValue = random(0, 257);  // Generate random number between 0-256
  char message[10];
  sprintf(message, "%d", randomValue);
  
  Serial.print("Publishing: ");
  Serial.println(message);

  client.publish("iot/intensity", message);
  delay(5000);  // Publish every 5 seconds
}
