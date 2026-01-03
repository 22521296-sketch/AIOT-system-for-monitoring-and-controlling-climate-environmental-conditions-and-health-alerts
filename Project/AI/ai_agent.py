#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECO MIND ENTERPRISE - AI AGENT CONTROLLER v2.0
=============================================================================
Author:      Google Gemini (AI Architect)
Description: Hệ thống AI trung tâm điều khiển môi trường thông minh (AIoT).
             Hỗ trợ học máy (Machine Learning), Xử lý ngôn ngữ tự nhiên (NLP),
             và Điều khiển thiết bị thời gian thực qua giao thức HTTP/Socket.IO.

Architecture:
    [Web Browser] <--(Socket.IO)--> [AI Agent (Python)] --(HTTP GET)--> [ESP32 Web Server]
                                          |
                                    [Data Engine]
                                   (CSV/Excel/ML)

Capabilities:
    1.  Multi-threaded Sensor Polling (Thu thập dữ liệu thời gian thực).
    2.  Dynamic Dataset Training (Học từ file CSV người dùng upload).
    3.  Personality Chat Engine (Chatbot có tính cách).
    4.  Direct Device Control (Điều khiển Quạt, Sưởi, Phun sương).
    5.  Fail-safe Mechanisms (Cơ chế an toàn khi mất kết nối).

Dependencies:
    pip install aiohttp python-socketio pandas numpy scikit-learn openpyxl colorama requests

Notes:
    - Fan 1: Quạt Mát (Cooling Fan)
    - Fan 2: Quạt Hút (Exhaust Fan)
=============================================================================
"""

import asyncio
import json
import logging
import os
import sys
import random
import io
import time
import math
import pickle
import glob

from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

# --- KIỂM TRA THƯ VIỆN ---
try:
    import aiohttp
    from aiohttp import web
    import socketio
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from colorama import init, Fore, Style, Back
except ImportError as e:
    print(f"CRITICAL ERROR: Thiếu thư viện hệ thống. Chi tiết: {e}")
    print("Vui lòng chạy lệnh: pip install aiohttp python-socketio pandas numpy scikit-learn openpyxl colorama requests")
    sys.exit(1)

# Khởi tạo màu sắc cho Terminal
init(autoreset=True)

# =============================================================================
# 1. SYSTEM CONFIGURATION (CẤU HÌNH HỆ THỐNG)
# =============================================================================

@dataclass
class SystemConfig:
    """
    Lớp cấu hình trung tâm. Chứa tất cả các tham số vận hành.
    """
    # --- MẠNG (NETWORK) ---
    HOST: str = "0.0.0.0"
    PORT: int = 3000
    
    # [QUAN TRỌNG] ĐỊA CHỈ IP CỦA ESP32
    # Bạn phải thay đổi địa chỉ này trùng với IP hiển thị trên Serial Monitor của ESP32
    ESP32_BASE_URL: str = "http://192.168.1.18"
    
    # --- THỜI GIAN (TIMING) ---
    POLLING_INTERVAL: int = 4         # Chu kỳ đọc cảm biến (giây)
    RETRY_DELAY: int = 5              # Thời gian chờ khi mất kết nối (giây)
    AUTOMATION_COOLDOWN: int = 10     # Thời gian nghỉ giữa các lần ra lệnh tự động (giây)

    # --- HỌC MÁY (MACHINE LEARNING) ---
    DATASET_DIR: str = "ai_memory"    # Thư mục lưu dữ liệu
    MODEL_PATH: str = "brain.pkl"     # Đường dẫn lưu Model (nếu cần)
    MIN_SAMPLES_FOR_TRAIN: int = 5    # Số lượng mẫu tối thiểu để kích hoạt học

    # --- NGƯỠNG MÔI TRƯỜNG (THRESHOLDS) ---
    # Nhiệt độ (Độ C)
    TEMP_COLD_LIMIT: float = 20.0     # Dưới mức này -> Lạnh
    TEMP_IDEAL_MIN: float = 22.0
    TEMP_IDEAL_MAX: float = 28.0
    TEMP_HOT_LIMIT: float = 30.0      # Trên mức này -> Nóng (Bật Fan 1)
    TEMP_EXTREME_LIMIT: float = 33.0  # Trên mức này -> Cực nóng (Bật Fan 2)

    # Độ ẩm (%)
    HUM_DRY_LIMIT: float = 50.0       # Dưới mức này -> Khô (Bật Phun sương)
    HUM_WET_LIMIT: float = 80.0       # Trên mức này -> Ẩm (Tắt Phun sương)

CONFIG = SystemConfig()

# =============================================================================
# 2. DATA MODELS & ENUMS (MÔ HÌNH DỮ LIỆU)
# =============================================================================

class DeviceType(Enum):
    FAN_COOLING = "fan1"  # Quạt Mát
    FAN_EXHAUST = "fan2"  # Quạt Hút
    MIST = "mist"         # Phun Sương
    HEATER = "heater"     # Sưởi

class DeviceAction(Enum):
    ON = "on"
    OFF = "off"

@dataclass
class SensorData:
    """Cấu trúc dữ liệu cảm biến tại một thời điểm."""
    temperature: float
    humidity: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        return {
            "temperature": self.temperature,
            "humidity": self.humidity,
            "timestamp": self.timestamp.strftime("%H:%M:%S")
        }

@dataclass
class CommandSignal:
    """Lệnh điều khiển được sinh ra bởi AI."""
    device: DeviceType
    action: DeviceAction
    reason: str
    priority: int = 1  # 1: Normal, 2: High, 3: Critical

# =============================================================================
# 3. LOGGING SYSTEM (HỆ THỐNG GHI LOG)
# =============================================================================

class LogSystem:
    """Hệ thống ghi log chuyên nghiệp với màu sắc."""
    @staticmethod
    def info(msg: str):
        print(f"{Fore.CYAN}[INFO]    {Style.RESET_ALL} {datetime.now().strftime('%H:%M:%S')} | {msg}")

    @staticmethod
    def success(msg: str):
        print(f"{Fore.GREEN}[SUCCESS] {Style.RESET_ALL} {datetime.now().strftime('%H:%M:%S')} | {msg}")

    @staticmethod
    def warning(msg: str):
        print(f"{Fore.YELLOW}[WARNING] {Style.RESET_ALL} {datetime.now().strftime('%H:%M:%S')} | {msg}")

    @staticmethod
    def error(msg: str):
        print(f"{Fore.RED}[ERROR]   {Style.RESET_ALL} {datetime.now().strftime('%H:%M:%S')} | {msg}")

    @staticmethod
    def ai(msg: str):
        print(f"{Fore.MAGENTA}[BRAIN]   {Style.RESET_ALL} {datetime.now().strftime('%H:%M:%S')} | {msg}")

    @staticmethod
    def network(msg: str):
        print(f"{Fore.BLUE}[NETWORK] {Style.RESET_ALL} {datetime.now().strftime('%H:%M:%S')} | {msg}")

# =============================================================================
# 4. DATA SCIENCE ENGINE (BỘ NÃO HỌC MÁY)
# =============================================================================

class EcoBrain:
    """
    Class chịu trách nhiệm xử lý dữ liệu CSV/Excel và huấn luyện mô hình.
    Cải tiến: Tự động chuẩn hóa tên cột và tối ưu hóa tham số học.
    """
    def __init__(self):
        # Tăng n_estimators để cây quyết định ổn định hơn
        self.regressor = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_stats = {}
        
        if not os.path.exists(CONFIG.DATASET_DIR):
            os.makedirs(CONFIG.DATASET_DIR)
            
        self.load_model()

    def ingest_data(self, filename: str, content: str) -> Dict[str, Any]:
        """Nhận dữ liệu upload và lưu file."""
        LogSystem.ai(f"Đang tiếp nhận dữ liệu mới: {filename}...")
        try:
            file_path = os.path.join(CONFIG.DATASET_DIR, filename)
            if isinstance(content, str):
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
            else:
                with open(file_path, "wb") as f:
                    f.write(content)
            return self.train_model()
        except Exception as e:
            LogSystem.error(f"Lỗi khi lưu dữ liệu: {str(e)}")
            return {"success": False, "message": f"Lỗi nội bộ: {str(e)}"}

    def train_model(self) -> Dict[str, Any]:
        """
        Hàm huấn luyện: Ưu tiên sử dụng cột 'Next_Temp' có sẵn trong CSV.
        """
        LogSystem.ai("Bắt đầu quy trình tái huấn luyện (Retraining)...")
        
        all_dfs = []
        files = glob.glob(os.path.join(CONFIG.DATASET_DIR, "*.*"))
        
        if not files:
            return {"success": False, "message": "Chưa có dữ liệu trong bộ nhớ."}

        for file_path in files:
            try:
                if file_path.endswith(".csv") or file_path.endswith(".txt"):
                    df = pd.read_csv(file_path)
                    all_dfs.append(df)
                elif "xls" in file_path:
                    df = pd.read_excel(file_path)
                    all_dfs.append(df)
            except Exception as e:
                LogSystem.warning(f"Không thể đọc file {file_path}: {e}")

        if not all_dfs:
            return {"success": False, "message": "Không đọc được dữ liệu hợp lệ nào."}

        full_df = pd.concat(all_dfs, ignore_index=True)

        # Làm sạch dữ liệu
        clean_df = self._clean_data(full_df)
        
        if len(clean_df) < CONFIG.MIN_SAMPLES_FOR_TRAIN:
            return {"success": False, "message": f"Dữ liệu quá ít (< {CONFIG.MIN_SAMPLES_FOR_TRAIN} mẫu)."}

        # --- LOGIC QUAN TRỌNG: XÁC ĐỊNH LABEL (Y) ---
        # Nếu trong file CSV đã có cột 'Next_Temp' (như file dataset của bạn), ta dùng nó luôn.
        # Nếu không có, ta mới dùng hàm shift(-1) để tự sinh dữ liệu.
        if 'Next_Temp' in clean_df.columns:
            # Xóa các dòng mà Next_Temp bị rỗng (nếu có)
            clean_df.dropna(subset=['Next_Temp'], inplace=True)
            LogSystem.ai("Phát hiện dữ liệu chuẩn (có Next_Temp). Đang học theo file...")
        else:
            # Tự động tính Next_Temp bằng cách lấy dòng tiếp theo
            clean_df['Next_Temp'] = clean_df['Temperature'].shift(-1)
            clean_df.dropna(inplace=True)
            LogSystem.ai("Dữ liệu thô (chưa có Next_Temp). Đang tự tính toán dòng thời gian...")

        # Training
        try:
            X = clean_df[['Temperature', 'Humidity']]
            y = clean_df['Next_Temp']
            
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            self.regressor.fit(X_scaled, y)
            self.is_trained = True
            self.save_model()

            stats = {
                "total_files": len(files),
                "total_samples": len(clean_df),
                "avg_temp": round(clean_df['Temperature'].mean(), 2),
                "score": round(self.regressor.score(X_scaled, y), 4)
            }
            self.training_stats = stats
            LogSystem.success(f"Training hoàn tất! R2 Score: {stats['score']}")
            return {"success": True, "message": "AI đã cập nhật kiến thức thành công.", "stats": stats}

        except Exception as e:
            LogSystem.error(f"Lỗi Training: {e}")
            return {"success": False, "message": str(e)}

        # --- FEATURE ENGINEERING (TẠO ĐẶC TRƯNG ĐỂ HỌC) ---
        # AI học: (Temp hiện tại, Hum hiện tại) -> (Temp tương lai)
        # Shift(-1) nghĩa là lấy giá trị của dòng tiếp theo làm mục tiêu (Label)
        clean_df['Target_Next_Temp'] = clean_df['Temperature'].shift(-1)
        
        # Loại bỏ dòng cuối cùng (vì không có dữ liệu tương lai)
        clean_df.dropna(inplace=True)

        try:
            X = clean_df[['Temperature', 'Humidity']]
            y = clean_df['Target_Next_Temp']
            
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            self.regressor.fit(X_scaled, y)
            self.is_trained = True
            self.save_model()

            stats = {
                "total_samples": len(clean_df),
                "accuracy_score": round(self.regressor.score(X_scaled, y) * 100, 2), # % độ chính xác
                "message": f"Đã học xong quy luật biến đổi nhiệt độ từ {len(clean_df)} mẫu dữ liệu."
            }
            LogSystem.success(f"Training xong! Độ chính xác mô hình: {stats['accuracy_score']}%")
            return {"success": True, "message": stats['message'], "stats": stats}

        except Exception as e:
            LogSystem.error(f"Lỗi Training: {e}")
            return {"success": False, "message": str(e)}

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Chuẩn hóa tên cột: Xóa khoảng trắng và đưa về chữ thường để dễ xử lý
        df.columns = [c.strip().lower() for c in df.columns]
        
        # 2. Map tên cột từ file CSV của bạn sang tên chuẩn của hệ thống
        rename_map = {
            'temp': 'Temperature',
            'hum': 'Humidity',
            'next_temp': 'Next_Temp',  # Cột quan trọng từ file của bạn
            't': 'Temperature',
            'h': 'Humidity'
        }
        df.rename(columns=rename_map, inplace=True)

        # 3. Kiểm tra xem có đủ dữ liệu đầu vào không
        if 'Temperature' not in df.columns or 'Humidity' not in df.columns:
            return pd.DataFrame()

        # 4. Ép kiểu dữ liệu sang số (tránh lỗi nếu file có lẫn chữ cái lạ)
        df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
        df['Humidity'] = pd.to_numeric(df['Humidity'], errors='coerce')
        
        # Nếu có cột Next_Temp, cũng ép kiểu luôn
        if 'Next_Temp' in df.columns:
            df['Next_Temp'] = pd.to_numeric(df['Next_Temp'], errors='coerce')

        # 5. Lọc bỏ dữ liệu nhiễu (NaN)
        df.dropna(subset=['Temperature', 'Humidity'], inplace=True)
        
        # Lọc bỏ các giá trị cảm biến vô lý (ví dụ: nhiệt độ 1000 độ)
        df = df[(df['Temperature'] > -10) & (df['Temperature'] < 60)]
        df = df[(df['Humidity'] > 0) & (df['Humidity'] <= 100)]
        
        return df

    def predict_next(self, current_temp, current_hum):
        """Trả về: (Nhiệt độ dự báo, Lời khuyên)"""
        if not self.is_trained:
            return current_temp, "Chưa có kiến thức"
        
        try:
            # --- SỬA LỖI TẠI ĐÂY ---
            # Thay vì đưa list [[temp, hum]], ta tạo DataFrame có tên cột khớp với lúc train
            input_data = pd.DataFrame(
                [[current_temp, current_hum]], 
                columns=['Temperature', 'Humidity']
            )
            
            # Giờ thì transform sẽ không còn báo Warning nữa
            X_new = self.scaler.transform(input_data)
            pred_temp = self.model.predict(X_new)[0]
            
            # --- Phần dưới giữ nguyên ---
            delta = pred_temp - current_temp
            
            trend = "Ổn định"
            if delta > 0.15: trend = f"Tăng nhẹ (+{delta:.2f})"
            if delta > 0.5: trend = f"TĂNG MẠNH (+{delta:.2f})"
            if delta < -0.15: trend = f"Giảm nhẹ ({delta:.2f})"
            
            return pred_temp, trend
        except Exception as e:
            # Log.error(f"Lỗi dự báo: {e}") # Có thể bỏ comment để debug
            return current_temp, "Lỗi"

    # (Giữ nguyên các hàm save_model, load_model cũ)
    def save_model(self):
        try:
            payload = {"model": self.regressor, "scaler": self.scaler, "stats": self.training_stats, "timestamp": datetime.now()}
            with open(CONFIG.MODEL_PATH, "wb") as f: pickle.dump(payload, f)
        except Exception as e: LogSystem.error(f"Lỗi lưu model: {e}")

    def load_model(self):
        if os.path.exists(CONFIG.MODEL_PATH):
            try:
                with open(CONFIG.MODEL_PATH, "rb") as f:
                    payload = pickle.load(f)
                    self.regressor = payload["model"]
                    self.scaler = payload["scaler"]
                    self.training_stats = payload.get("stats", {})
                    self.is_trained = True
                LogSystem.success("Đã khôi phục não bộ AI.")
            except: pass
# =============================================================================
# 5. PERSONALITY ENGINE (TRÍ TUỆ CẢM XÚC & NLP)
# =============================================================================

class PersonalityEngine:
    """
    Quản lý tính cách và phản hồi chat của AI.
    Tên: EcoBot
    Tính cách: Thân thiện, chuyên nghiệp, quan tâm môi trường.
    """
    def __init__(self):
        self.name = "EcoBot"
        self.context = []
    
    def process_message(self, message: str, env: SensorData, brain: EcoBrain) -> Tuple[str, Optional[CommandSignal]]:
        """
        Phân tích tin nhắn người dùng -> Trả về (Câu trả lời, Lệnh điều khiển nếu có).
        """
        msg = message.lower().strip()
        command = self._extract_command(msg)
        
        # 1. Nếu có lệnh điều khiển rõ ràng
        if command:
            device_name_vn = self._get_device_name_vn(command.device)
            action_vn = "BẬT" if command.action == DeviceAction.ON else "TẮT"
            return f"Đã rõ! Tôi đang gửi lệnh {action_vn} {device_name_vn} ngay đây.", command

        # 2. Các câu hỏi thông thường
        if any(x in msg for x in ["chào", "hi", "hello", "alo"]):
            return "Xin chào! Tôi là AI quản lý hệ thống. Môi trường hôm nay thế nào?", None
        
        if any(x in msg for x in ["nhiệt độ", "độ ẩm", "tình hình", "báo cáo"]):
            trend = brain.predict_trend(env.temperature, env.humidity)
            return (f"Báo cáo: Nhiệt độ {env.temperature}°C, Độ ẩm {env.humidity}%. "
                    f"({trend})"), None

        if "nóng" in msg:
            return "Tôi thấy nhiệt độ đang cao. Bạn có muốn tôi bật Quạt Mát (Fan 1) không?", None
        
        if "lạnh" in msg:
            return "Trời hơi lạnh. Tôi có thể bật sưởi nếu bạn cần.", None

        # 3. Mặc định
        return "Tôi đang lắng nghe. Bạn có thể ra lệnh như 'Bật quạt mát' hoặc hỏi 'Nhiệt độ bao nhiêu'.", None

    def _extract_command(self, msg: str) -> Optional[CommandSignal]:
        """Logic NLP đơn giản để trích xuất ý định (Intent Recognition)."""
        action = None
        if "bật" in msg or "mở" in msg or "kích hoạt" in msg:
            action = DeviceAction.ON
        elif "tắt" in msg or "ngừng" in msg or "dừng" in msg:
            action = DeviceAction.OFF
            
        if not action: return None

        # Mapping thiết bị
        # Logic: Fan 1 = Mát, Fan 2 = Hút
        if "quạt mát" in msg or ("quạt" in msg and "hút" not in msg and "2" not in msg):
            return CommandSignal(DeviceType.FAN_COOLING, action, "User request")
        
        if "quạt hút" in msg or "thông gió" in msg or "quạt 2" in msg:
            return CommandSignal(DeviceType.FAN_EXHAUST, action, "User request")
            
        if "sưởi" in msg or "lò sưởi" in msg:
            return CommandSignal(DeviceType.HEATER, action, "User request")
            
        if "phun sương" in msg or "ẩm" in msg:
            return CommandSignal(DeviceType.MIST, action, "User request")
            
        return None

    def _get_device_name_vn(self, device: DeviceType) -> str:
        map_vn = {
            DeviceType.FAN_COOLING: "Quạt Mát",
            DeviceType.FAN_EXHAUST: "Quạt Hút",
            DeviceType.MIST: "Phun Sương",
            DeviceType.HEATER: "Sưởi"
        }
        return map_vn.get(device, "Thiết bị")

# =============================================================================
# 6. AUTOMATION CONTROLLER (BỘ ĐIỀU KHIỂN TỰ ĐỘNG)
# =============================================================================

class AutomationController:
    """
    Điều khiển thiết bị dựa trên cả ngưỡng (Threshold) và dự báo AI (Prediction).
    """
    def __init__(self):
        self.device_states = {
            DeviceType.FAN_COOLING: DeviceAction.OFF,
            DeviceType.FAN_EXHAUST: DeviceAction.OFF,
            DeviceType.MIST: DeviceAction.OFF,
            DeviceType.HEATER: DeviceAction.OFF
        }
        self.last_run = datetime.now()

    def process(self, env: SensorData, brain: EcoBrain) -> List[CommandSignal]:
        commands = []
        if (datetime.now() - self.last_run).total_seconds() < CONFIG.AUTOMATION_COOLDOWN:
            return commands

        # 1. Lấy dự báo từ AI
        pred_temp, trend_text = brain.predict_next(env.temperature, env.humidity)
        
        # LOGIC AI: Pre-emptive Cooling (Làm mát đón đầu)
        # Nếu nhiệt độ hiện tại chưa tới mức nóng (VD: 29 độ), 
        # nhưng AI đoán sắp tới sẽ lên 30.5 độ -> Bật quạt ngay từ bây giờ.
        ai_warning_hot = (env.temperature >= 28.0 and pred_temp > CONFIG.TEMP_HOT_LIMIT)

        # --- QUẢN LÝ NHIỆT ĐỘ ---
        
        # BẬT QUẠT MÁT KHI: Nóng thực tế HOẶC AI cảnh báo sắp nóng
        if env.temperature > CONFIG.TEMP_HOT_LIMIT or ai_warning_hot:
            reason = "Quá nóng (>30°C)" if env.temperature > CONFIG.TEMP_HOT_LIMIT else f"AI dự báo nhiệt tăng lên {pred_temp:.1f}°C"
            
            if self.device_states[DeviceType.FAN_COOLING] == DeviceAction.OFF:
                commands.append(CommandSignal(DeviceType.FAN_COOLING, DeviceAction.ON, reason))
                
            # Cực nóng -> Bật thêm hút
            if env.temperature > CONFIG.TEMP_EXTREME_LIMIT or pred_temp > CONFIG.TEMP_EXTREME_LIMIT:
                if self.device_states[DeviceType.FAN_EXHAUST] == DeviceAction.OFF:
                    commands.append(CommandSignal(DeviceType.FAN_EXHAUST, DeviceAction.ON, "Cảnh báo nhiệt độ cao"))

        # BẬT SƯỞI
        elif env.temperature < CONFIG.TEMP_COLD_LIMIT:
            if self.device_states[DeviceType.HEATER] == DeviceAction.OFF:
                commands.append(CommandSignal(DeviceType.HEATER, DeviceAction.ON, "Nhiệt độ thấp"))
            if self.device_states[DeviceType.FAN_COOLING] == DeviceAction.ON:
                commands.append(CommandSignal(DeviceType.FAN_COOLING, DeviceAction.OFF, "Tắt quạt để sưởi ấm"))

        # TRẠNG THÁI LÝ TƯỞNG -> Tắt hết để tiết kiệm điện
        elif CONFIG.TEMP_IDEAL_MIN <= env.temperature <= CONFIG.TEMP_IDEAL_MAX and not ai_warning_hot:
            # Chỉ tắt nếu AI không cảnh báo nóng
            if self.device_states[DeviceType.FAN_COOLING] == DeviceAction.ON:
                commands.append(CommandSignal(DeviceType.FAN_COOLING, DeviceAction.OFF, "Môi trường ổn định"))
            if self.device_states[DeviceType.FAN_EXHAUST] == DeviceAction.ON:
                commands.append(CommandSignal(DeviceType.FAN_EXHAUST, DeviceAction.OFF, "Môi trường ổn định"))
            if self.device_states[DeviceType.HEATER] == DeviceAction.ON:
                commands.append(CommandSignal(DeviceType.HEATER, DeviceAction.OFF, "Môi trường ổn định"))

        # --- QUẢN LÝ ĐỘ ẨM (Giữ nguyên logic cũ) ---
        if env.humidity < CONFIG.HUM_DRY_LIMIT:
            if self.device_states[DeviceType.MIST] == DeviceAction.OFF:
                commands.append(CommandSignal(DeviceType.MIST, DeviceAction.ON, "Độ ẩm thấp"))
        elif env.humidity > CONFIG.HUM_WET_LIMIT:
            if self.device_states[DeviceType.MIST] == DeviceAction.ON:
                commands.append(CommandSignal(DeviceType.MIST, DeviceAction.OFF, "Độ ẩm cao"))

        # Cập nhật state
        for cmd in commands:
            self.device_states[cmd.device] = cmd.action
        
        if commands:
            self.last_run = datetime.now()
            
        return commands

    def update_state_manual(self, device: DeviceType, action: DeviceAction):
        self.device_states[device] = action

# =============================================================================
# 7. MAIN SERVER APPLICATION (ỨNG DỤNG SERVER CHÍNH)
# =============================================================================

class EcoSmartServer:
    """
    Lớp trung tâm kết nối mọi thành phần.
    """
    def __init__(self):
        # Setup Server Web (Socket.IO + Aiohttp)
        self.sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
        self.app = web.Application()
        self.sio.attach(self.app)
        
        # Khởi tạo các module con
        self.brain = EcoBrain()
        self.persona = PersonalityEngine()
        self.controller = AutomationController()
        
        # State
        self.current_env = SensorData(25.0, 60.0) # Giá trị mặc định
        self.is_running = True

        # Đăng ký các sự kiện
        self._register_routes()
        self._register_socket_events()

    def _register_routes(self):
        # Route kiểm tra sức khỏe server
        self.app.router.add_get('/', self._handle_index)

    def _register_socket_events(self):
        @self.sio.event
        async def connect(sid, environ):
            LogSystem.network(f"Client Web đã kết nối: {sid}")
            await self.sio.emit('ai_chat_reply', {'reply': f"AI Online. {self.persona.process_message('hello', self.current_env, self.brain)[0]}"}, room=sid)

        @self.sio.event
        async def disconnect(sid):
            LogSystem.network(f"Client Web ngắt kết nối: {sid}")

        @self.sio.event
        async def user_chat_message(sid, data):
            """Xử lý tin nhắn chat từ người dùng."""
            raw_msg = data.get('message', '')
            LogSystem.info(f"User Chat: {raw_msg}")
            
            # 1. NLP xử lý
            reply_text, command = self.persona.process_message(raw_msg, self.current_env, self.brain)
            
            # 2. Thực thi lệnh nếu có
            if command:
                success = await self.execute_device_command(command)
                if success:
                    reply_text += " (Đã thực hiện ✅)"
                else:
                    reply_text += " (Lỗi kết nối thiết bị ❌)"

            # 3. Phản hồi
            await self.sio.emit('ai_chat_reply', {'reply': reply_text}, room=sid)

        @self.sio.event
        async def upload_dataset(sid, data):
            """Xử lý sự kiện upload file CSV."""
            filename = data.get('filename', 'unknown.csv')
            content = data.get('content', '')
            
            # Chạy xử lý dữ liệu trong Thread riêng để không chặn server
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.brain.ingest_data, filename, content)
            
            await self.sio.emit('ai_chat_reply', {'reply': result['message']}, room=sid)
            if result['success']:
                # Gửi sự kiện cập nhật bảng dữ liệu trên web (nếu có)
                await self.sio.emit('update_dataset_row', {'filename': filename, 'stats': result['stats']}, room=sid)

    async def _handle_index(self, request):
        return web.Response(text="Eco Smart AI Agent is Running...")

    async def execute_device_command(self, cmd: CommandSignal) -> bool:
        """
        Gửi lệnh điều khiển trực tiếp tới ESP32 thông qua HTTP Request.
        Endpoint: /control?device=xxx&state=on/off
        """
        LogSystem.ai(f"RA LỆNH: {cmd.device.value} -> {cmd.action.value} ({cmd.reason})")
        
        # 1. Cập nhật UI trên Web (thông qua Socket)
        await self.sio.emit('ai_command', {
            'device': cmd.device.value,
            'action': cmd.action.value
        })
        
        # 2. Cập nhật trạng thái nội bộ của Controller
        self.controller.update_state_manual(cmd.device, cmd.action)

        # 3. Gửi Request vật lý tới ESP32
        target_url = f"{CONFIG.ESP32_BASE_URL}/control"
        params = {
            'device': cmd.device.value,
            'state': cmd.action.value
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(target_url, params=params, timeout=3) as resp:
                    if resp.status == 200:
                        LogSystem.success(f"ESP32 xác nhận lệnh {cmd.device.value} {cmd.action.value}")
                        return True
                    else:
                        LogSystem.warning(f"ESP32 trả về mã lỗi: {resp.status}")
                        return False
        except Exception as e:
            LogSystem.error(f"Không thể kết nối tới ESP32 tại {CONFIG.ESP32_BASE_URL}. Lỗi: {e}")
            return False

    async def background_sensor_polling(self):
        LogSystem.info("Khởi động dịch vụ giám sát cảm biến...")
        async with aiohttp.ClientSession() as session:
            while self.is_running:
                start_time = time.time()
                try:
                    async with session.get(f"{CONFIG.ESP32_BASE_URL}/status", timeout=CONFIG.POLLING_INTERVAL) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self.current_env.temperature = float(data.get('temp', 0))
                            self.current_env.humidity = float(data.get('hum', 0))
                            self.current_env.timestamp = datetime.now()
                except Exception:
                    pass

                # --- SỬA ĐỔI Ở ĐÂY ---
                # Truyền cả object 'brain' vào process để controller tham khảo ý kiến AI
                auto_commands = self.controller.process(self.current_env, self.brain)
                # ---------------------
                
                for cmd in auto_commands:
                    await self.execute_device_command(cmd)
                    await self.sio.emit('ai_chat_reply', {
                        'reply': f"🤖 Tự động: {cmd.reason} -> {cmd.action.value.upper()} {cmd.device.value}."
                    })

                elapsed = time.time() - start_time
                await asyncio.sleep(max(0, CONFIG.POLLING_INTERVAL - elapsed))

    async def start(self):
        """Khởi động toàn bộ hệ thống."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, CONFIG.HOST, CONFIG.PORT)
        
        print(f"\n{Back.BLUE}{Fore.WHITE} ECO MIND AI SYSTEM - STARTED {Style.RESET_ALL}")
        print(f"{Fore.CYAN}Server listening at: http://localhost:{CONFIG.PORT}")
        print(f"{Fore.CYAN}ESP32 Endpoint:    {CONFIG.ESP32_BASE_URL}")
        print(f"{Fore.CYAN}Data Storage:      {CONFIG.DATASET_DIR}")
        print("-" * 50)

        # Chạy tác vụ nền
        asyncio.create_task(self.background_sensor_polling())
        
        # Start Server
        await site.start()
        
        # Giữ process sống
        while True:
            await asyncio.sleep(3600)

# =============================================================================
# 8. ENTRY POINT (ĐIỂM KHỞI CHẠY)
# =============================================================================

if __name__ == "__main__":
    # Fix lỗi aiohttp trên Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        server = EcoSmartServer()
        asyncio.run(server.start())
    except KeyboardInterrupt:
        LogSystem.warning("Server đang dừng theo yêu cầu người dùng...")
    except Exception as e:
        LogSystem.error(f"Lỗi nghiêm trọng không mong muốn: {e}")
