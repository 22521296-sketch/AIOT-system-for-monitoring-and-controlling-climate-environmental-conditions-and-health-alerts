#ifndef ACTUATOR_CTRL_H
#define ACTUATOR_CTRL_H

#include <Arduino.h>

// ==========================================
// 1. CẤU HÌNH CHÂN GPIO (PINOUT)
// ==========================================
// Sử dụng các chân an toàn cho ESP32 (Tránh chân 6-11, 0, 1, 3, 34-39)
#define FAN1_PIN    6  // Quạt 1
#define FAN2_PIN    17  // Quạt 2
#define MIST_PIN    16  // Phun sương
#define HEATER_PIN  7  // Sấy / Đèn

// ==========================================
// 2. CẤU HÌNH LOGIC RELAY
// ==========================================
// Nếu Relay của bạn là loại "Mức Thấp" (Kích 0V là bật): để là LOW
// Nếu Relay của bạn là loại "Mức Cao" (Kích 3.3V là bật): đổi thành HIGH
#define RELAY_ON    LOW  
#define RELAY_OFF   HIGH 

// ==========================================
// 3. CÁC HÀM ĐIỀU KHIỂN
// ==========================================

// Hàm bật/tắt Quạt 1
void controlFan1(bool turnOn) {
    digitalWrite(FAN1_PIN, turnOn ? RELAY_ON : RELAY_OFF);
    Serial.print(">> Fan 1: ");
    Serial.println(turnOn ? "ON" : "OFF");
}

// Hàm bật/tắt Quạt 2
void controlFan2(bool turnOn) {
    digitalWrite(FAN2_PIN, turnOn ? RELAY_ON : RELAY_OFF);
    Serial.print(">> Fan 2: ");
    Serial.println(turnOn ? "ON" : "OFF");
}

// Hàm bật/tắt Phun sương
void controlMist(bool turnOn) {
    digitalWrite(MIST_PIN, turnOn ? RELAY_ON : RELAY_OFF);
    Serial.print(">> Mist: ");
    Serial.println(turnOn ? "ON" : "OFF");
}

// Hàm bật/tắt Sấy (Heater)
void controlHeater(bool turnOn) {
    digitalWrite(HEATER_PIN, turnOn ? RELAY_ON : RELAY_OFF);
    Serial.print(">> Heater: ");
    Serial.println(turnOn ? "ON" : "OFF");
}

// ==========================================
// 4. HÀM KHỞI TẠO (GỌI TRONG SETUP)
// ==========================================
void setupActuators() {
    // Cài đặt chân là Output
    pinMode(FAN1_PIN, OUTPUT);
    pinMode(FAN2_PIN, OUTPUT);
    pinMode(MIST_PIN, OUTPUT);
    pinMode(HEATER_PIN, OUTPUT);

    // Mặc định tắt hết khi khởi động để an toàn
    digitalWrite(FAN1_PIN, RELAY_OFF);
    digitalWrite(FAN2_PIN, RELAY_OFF);
    digitalWrite(MIST_PIN, RELAY_OFF);
    digitalWrite(HEATER_PIN, RELAY_OFF);

    Serial.println("--- Da khoi tao thiet bi (Actuators) ---");
}

#endif