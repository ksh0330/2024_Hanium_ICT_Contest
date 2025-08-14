#include <WiFi.h>
#include <Adafruit_NeoPixel.h>

#define LED_PIN 8      // LED 핀 번호
#define BUZZER_PIN 9   // 부저 핀 번호
#define NUM_PIXELS 7   // NeoPixel 사용 개수
#define PROCESS_DELAY 2000 // 메시지 처리 후 대기 시간 (밀리초 단위)

const char* ssid = "***"; // Wi-Fi SSID
const char* password = "***"; // Wi-Fi Password

const char* host = "192.168.0.*"; // 서버 IP 주소
const int port = 8878; // 서버 포트

WiFiClient client;
Adafruit_NeoPixel strip = Adafruit_NeoPixel(NUM_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

String lastMessage = "";  // 마지막으로 수신된 메시지
unsigned long lastProcessTime = 0; // 마지막으로 메시지를 처리한 시간

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
      String message = client.readStringUntil('\n');
      Serial.println("Message from server: " + message);

      // 이전 메시지와 다르고, 지정된 시간(PROCESS_DELAY) 이후에만 메시지 처리
      if (message != lastMessage && millis() - lastProcessTime > PROCESS_DELAY) {  
        lastMessage = message;
        handleServerMessage(message);
        lastProcessTime = millis();  // 마지막 처리 시간 갱신
      }
    }
  } else {
    Serial.println("Disconnected from server, attempting to reconnect...");
    if (client.connect(host, port)) {
      Serial.println("Reconnected to server");
    }
  }

  maintainState();  // 현재 상태 유지
}

void connectToWiFi() {
  Serial.print("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("Connected to WiFi");
  Serial.println("IP Address: ");
  Serial.println(WiFi.localIP());
}

void handleServerMessage(String message) {
  if (message.indexOf("안전") >= 0) {
    setLEDColor(strip.Color(0, 255, 0)); // G
    noTone(BUZZER_PIN);
  } else if (message.indexOf("주의") >= 0) {
    setLEDColor(strip.Color(255, 255, 0)); // Y
    noTone(BUZZER_PIN); 
  } else if (message.indexOf("위험") >= 0) {
    setLEDColor(strip.Color(255, 0, 0)); // R
    tone(BUZZER_PIN, 500); 
  }
}

void setLEDColor(uint32_t color) {
  for (int i = 0; i < NUM_PIXELS; i++) {
    strip.setPixelColor(i, color);
  }
  strip.show();
}

void maintainState() {
  // 현재 상태를 유지하기 위해 아무것도 하지 않음.
  // 이 함수는 단지 LED와 부저의 상태를 유지하기 위해 필요.
}
