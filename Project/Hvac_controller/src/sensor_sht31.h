#ifndef SENSOR_SHT31_H
#define SENSOR_SHT31_H

#include <Wire.h>
#include <Adafruit_SHT31.h>

// --- CẤU HÌNH I2C CHO ESP32-S3 (GIỮ NGUYÊN) ---
#define SHT31_SDA   18
#define SHT31_SCL   8

Adafruit_SHT31 sht31 = Adafruit_SHT31();

void setupSensor() {
    // Khởi tạo I2C
    Wire.begin(SHT31_SDA, SHT31_SCL, 100000);

    if (!sht31.begin(0x44)) {   
        Serial.println("[SHT31] Khong tim thay cam bien!");
    } else {
        Serial.println("[SHT31] Da khoi tao thanh cong.");
    }
}

// Hàm mới: Đọc dữ liệu và lưu vào biến tham chiếu
void readSHT31(float &t, float &h) {
    float temp_val = sht31.readTemperature();
    float hum_val = sht31.readHumidity();

    // Kiểm tra lỗi (NaN)
    if (!isnan(temp_val)) {
        t = temp_val;
    } else {
        Serial.println("[SHT31] Loi doc Nhiet do");
    }

    if (!isnan(hum_val)) {
        h = hum_val;
    } else {
        Serial.println("[SHT31] Loi doc Do am");
    }
}

#endif