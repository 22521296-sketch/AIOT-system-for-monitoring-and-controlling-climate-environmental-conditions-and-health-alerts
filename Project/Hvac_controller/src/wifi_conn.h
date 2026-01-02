#ifndef WIFI_CONN_H
#define WIFI_CONN_H

#include <WiFi.h>

void setupWiFi(const char* ssid, const char* pass) {
    Serial.print("[WiFi] Dang ket noi den: ");
    Serial.println(ssid);

    WiFi.mode(WIFI_STA); // Chế độ Station
    WiFi.begin(ssid, pass);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(5000);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n[WiFi] Da ket noi!");
        Serial.print("[WiFi] IP Address: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\n[WiFi] Ket noi that bai.");
    }
}

void checkWiFi() {
    // Nếu mất WiFi thì tự kết nối lại
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[WiFi] Mat ket noi! Dang ket noi lai...");
        WiFi.reconnect();
    }
}

#endif