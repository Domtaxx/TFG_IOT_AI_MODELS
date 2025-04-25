#include <WiFi.h>     // or <WiFi.h> for ESP32
#include <PubSubClient.h>

const char* ssid = "WQ Router";
const char* password = "UliMar17";
const char* mqtt_server = "raspberrypi.local";  // e.g., "192.168.1.100"

WiFiClient espClient;
PubSubClient client(espClient);

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("]: ");
  String message;
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  Serial.println(message);

  // Check if the received message is "blink"
  if (message == "blink") {
    // Blink the built-in LED
    digitalWrite(LED_BUILTIN, LOW);  // Turn on (note: on many boards LOW turns the LED on)
    delay(500);
    digitalWrite(LED_BUILTIN, HIGH); // Turn off
  }
}

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect("ESPClient")) {
      Serial.println("connected");
      // Once connected, subscribe to the blink topic
      client.subscribe("esp/blink");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" - try again in 5 seconds");
      delay(5000);
    }
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}
