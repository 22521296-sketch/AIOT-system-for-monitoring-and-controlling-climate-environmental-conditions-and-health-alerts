#include <Arduino.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <SPIFFS.h>
#include <HTTPClient.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// --- KHAI BÁO BIẾN TOÀN CỤC (Để WebServer.h nhìn thấy) ---
float temperature = 0.0;
float humidity = 0.0;

// Include các file header
#include "sensor_sht31.h"  // SHT31
#include "actuator_ctrl.h" // Điều khiển Relay
#include "web_server.h"    // Giao diện Web

WebSocketsClient webSocket;

// --- THÔNG TIN KẾT NỐI ---
const char* WIFI_SSID = "xxx";
const char* WIFI_PASS = "xxx";
const char* SERVER_IP = "xxxxxx"; 
const int SERVER_PORT = 8080;            

// Hàm xử lý WebSocket
void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch (type) {
    case WStype_DISCONNECTED:
      Serial.println("[WebSocket] Ngat ket noi!");            
      break;
    case WStype_CONNECTED:
      Serial.println("[WebSocket] Da ket noi Server!");
      webSocket.sendTXT("{\"type\":\"device_init\", \"id\":\"ESP32_S3_AGENT\"}");
      break;
    case WStype_TEXT: {
      String payload_str = String((char*)payload);
      Serial.printf("[WebSocket] Nhan: %s\n", payload_str.c_str());
      StaticJsonDocument<512> doc;
      deserializeJson(doc, payload_str);
      
      if (doc["command"] == "SET_DEVICE") {     
        const char* dev = doc["device"];
        const char* sta = doc["status"];
        bool on = (strcmp(sta, "ON") == 0);

        if (strcmp(dev, "FAN1") == 0) controlFan1(on);
        else if (strcmp(dev, "MIST") == 0) controlMist(on);
        else if (strcmp(dev, "FAN2") == 0) controlFan2(on);
        else if (strcmp(dev, "HEATER") == 0) controlHeater(on);
      }
      break;
    }
    case WStype_BIN: break; // Bỏ qua các case không dùng cho gọn
  }
}

void setup() {
  // 1. Chống Brownout 
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  
  Serial.begin(115200);
  delay(1000);

  // 2. Khởi tạo phần cứng
  setupActuators(); 
  setupSensor();    

  // 3. Kết nối WiFi
  Serial.print("Ket noi WiFi: ");
  Serial.println(WIFI_SSID);
  
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false); // Tắt tiết kiệm điện để WiFi ổn định
  WiFi.setTxPower(WIFI_POWER_8_5dBm); // Giảm công suất để tránh sụt áp
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  int tryCount = 0;
  while (WiFi.status() != WL_CONNECTED && tryCount < 20) {
      delay(500);
      Serial.print(".");
      tryCount++;
  }
  Serial.println("\nWiFi OK!");

  // 4. Khởi động Web Server & WebSocket
  setupWebServer(); 
  
  webSocket.begin(SERVER_IP, SERVER_PORT, "/");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000); 
}

void loop() {
  webSocket.loop(); 

  // Nếu mất WiFi thì thử kết nối lại
  if(WiFi.status() != WL_CONNECTED) {
      WiFi.reconnect();
  }
  
  // Gửi dữ liệu định kỳ (mỗi 2 giây)
  static unsigned long lastTime = 0;
  if (millis() - lastTime > 2000) {
    
    // 1. Đọc từ cảm biến cập nhật vào biến toàn cục
    readSHT31(temperature, humidity);

    // 2. In ra Serial để kiểm tra
    Serial.printf("Temp: %.1f C | Hum: %.1f %%\n", temperature, humidity);

    // 3. Gửi qua WebSocket (Tạo JSON từ biến toàn cục)
    String jsonString = "{\"type\":\"sensor_data\",\"temp\":" + String(temperature) + ",\"hum\":" + String(humidity) + "}";
    webSocket.sendTXT(jsonString);
    
    lastTime = millis();
  }

}

