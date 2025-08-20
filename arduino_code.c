#include <WiFi.h>
#include <Adafruit_NeoPixel.h>

#define LED_PIN 8            // LED pin
#define BUZZER_PIN 9         // Buzzer pin
#define NUM_PIXELS 7         // Number of NeoPixels
#define PROCESS_DELAY 2000   // Delay after processing a message (ms)

const char* ssid = "***";         // Wi-Fi SSID
const char* password = "***";     // Wi-Fi Password

const char* host = "192.168.0.*"; // Server IP address
const int port = 8878;            // Server port

WiFiClient client;
Adafruit_NeoPixel strip = Adafruit_NeoPixel(NUM_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

String lastMessage = "";               // Last received message
unsigned long lastProcessTime = 0;     // Last time a message was processed

void connectToWiFi() {
  Serial.print("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println(" connected");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

void setLEDColor(uint32_t color) {
  for (int i = 0; i < NUM_PIXELS; i++) {
    strip.setPixelColor(i, color);
  }
  strip.show();
}

void handleServerMessage(const String& message) {
  // Keep Korean status keywords for compatibility with the sender
  if (message.indexOf("안전") >= 0) {
    setLEDColor(strip.Color(0, 255, 0)); // Green
    noTone(BUZZER_PIN);
  } else if (message.indexOf("주의") >= 0) {
    setLEDColor(strip.Color(255, 255, 0)); // Yellow
    noTone(BUZZER_PIN);
  } else if (message.indexOf("위험") >= 0) {
    setLEDColor(strip.Color(255, 0, 0)); // Red
    tone(BUZZER_PIN, 500);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  strip.begin();
  strip.show(); // Initialize all pixels to 'off'

  connectToWiFi();

  if (client.connect(host, port)) {
    Serial.println("Connected to server");
  } else {
    Serial.println("Connection to server failed");
  }
}

void loop() {
  if (client.connected()) {
    if (client.available()) {
      String message = client.readStringUntil('\n'); // Expect newline-terminated lines
      message.trim();
      if (message.length() > 0) {
        Serial.println("Message from server: " + message);
      }

      // Process only when message changed and delay elapsed
      if (message != lastMessage && millis() - lastProcessTime > PROCESS_DELAY) {
        lastMessage = message;
        handleServerMessage(message);
        lastProcessTime = millis();
      }
    }
  } else {
    Serial.println("Disconnected from server, attempting to reconnect...");
    if (client.connect(host, port)) {
      Serial.println("Reconnected to server");
    }
    delay(250); // Throttle reconnect attempts slightly
  }
}
