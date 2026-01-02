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
    ESP32_BASE_URL: str = "http://192.168.1.3"
    
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
    Sử dụng RandomForestRegressor để dự đoán xu hướng.
    """
    def __init__(self):
        self.regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_stats = {}
        
        # Tạo thư mục lưu trữ nếu chưa có
        if not os.path.exists(CONFIG.DATASET_DIR):
            os.makedirs(CONFIG.DATASET_DIR)

    def ingest_data(self, filename: str, content: str) -> Dict[str, Any]:
        """
        Nhận dữ liệu thô (string), parse thành DataFrame và training.
        """
        LogSystem.ai(f"Đang phân tích file dữ liệu: {filename}...")
        
        try:
            # 1. Parsing Data
            df = self._parse_content(filename, content)
            if df is None:
                return {"success": False, "message": "Định dạng file không hợp lệ (Dùng CSV/Excel)."}

            # 2. Cleaning Data
            df = self._clean_data(df)
            if df.empty or len(df) < CONFIG.MIN_SAMPLES_FOR_TRAIN:
                return {"success": False, "message": f"Dữ liệu quá ít (< {CONFIG.MIN_SAMPLES_FOR_TRAIN} mẫu)."}

            # 3. Feature Engineering (Tạo dữ liệu để học)
            # Học mối quan hệ: (Temp hiện tại, Hum hiện tại) -> (Temp tương lai)
            df['Next_Temp'] = df['Temperature'].shift(-1)
            df.dropna(inplace=True)

            # 4. Training
            X = df[['Temperature', 'Humidity']]
            y = df['Next_Temp']
            
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            self.regressor.fit(X_scaled, y)
            self.is_trained = True

            # 5. Calculate Statistics
            stats = {
                "samples": len(df),
                "avg_temp": round(df['Temperature'].mean(), 2),
                "avg_hum": round(df['Humidity'].mean(), 2),
                "max_temp": df['Temperature'].max(),
                "correlation": round(df['Temperature'].corr(df['Humidity']), 2)
            }
            self.training_stats = stats
            
            LogSystem.success(f"Training hoàn tất! Đã học {len(df)} mẫu dữ liệu.")
            return {"success": True, "message": "AI đã học xong dữ liệu mới.", "stats": stats}

        except Exception as e:
            LogSystem.error(f"Lỗi trong quá trình học: {str(e)}")
            return {"success": False, "message": f"Lỗi nội bộ: {str(e)}"}

    def _parse_content(self, filename: str, content: str) -> Optional[pd.DataFrame]:
        try:
            if filename.endswith('.csv') or filename.endswith('.txt'):
                return pd.read_csv(io.StringIO(content))
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                # Xử lý binary stream giả lập cho Excel
                return pd.read_excel(io.BytesIO(content.encode('latin1')))
            return None
        except Exception:
            return None

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Chuẩn hóa tên cột
        df.columns = [c.strip().title() for c in df.columns]
        
        # Mapping tên cột thông dụng
        col_map = {
            'Temp': 'Temperature', 'T': 'Temperature', 'Nhietdo': 'Temperature',
            'Hum': 'Humidity', 'H': 'Humidity', 'Doam': 'Humidity'
        }
        df.rename(columns=col_map, inplace=True)
        
        # Kiểm tra cột bắt buộc
        if 'Temperature' not in df.columns or 'Humidity' not in df.columns:
            return pd.DataFrame()

        # Lọc nhiễu
        df = df[(df['Temperature'] > -10) & (df['Temperature'] < 60)]
        df = df[(df['Humidity'] > 0) & (df['Humidity'] <= 100)]
        
        return df

    def predict_trend(self, current_temp: float, current_hum: float) -> str:
        """Dự báo xu hướng nhiệt độ."""
        if not self.is_trained:
            return "Chưa có dữ liệu học."
        
        try:
            X_in = self.scaler.transform([[current_temp, current_hum]])
            pred_temp = self.regressor.predict(X_in)[0]
            delta = pred_temp - current_temp
            
            if delta > 0.15: return "Dự báo: Tăng nhiệt 📈"
            if delta < -0.15: return "Dự báo: Giảm nhiệt 📉"
            return "Dự báo: Ổn định ➡️"
        except:
            return "Lỗi dự đoán"

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
    Chịu trách nhiệm ra quyết định tự động dựa trên dữ liệu cảm biến.
    """
    def __init__(self):
        # Lưu trạng thái nội bộ để tránh spam lệnh
        self.device_states = {
            DeviceType.FAN_COOLING: DeviceAction.OFF,
            DeviceType.FAN_EXHAUST: DeviceAction.OFF,
            DeviceType.MIST: DeviceAction.OFF,
            DeviceType.HEATER: DeviceAction.OFF
        }
        self.last_run = datetime.now()

    def process(self, env: SensorData, trend: str) -> List[CommandSignal]:
        commands = []
        
        # Kiểm tra cooldown (tránh bật tắt liên tục gây hại thiết bị)
        if (datetime.now() - self.last_run).total_seconds() < CONFIG.AUTOMATION_COOLDOWN:
            return commands

        # --- LOGIC 1: QUẢN LÝ NHIỆT ĐỘ ---
        
        # TRƯỜNG HỢP: QUÁ NÓNG (> 30 độ)
        if env.temperature > CONFIG.TEMP_HOT_LIMIT:
            # Bật Quạt Mát (Fan 1)
            if self.device_states[DeviceType.FAN_COOLING] == DeviceAction.OFF:
                commands.append(CommandSignal(DeviceType.FAN_COOLING, DeviceAction.ON, "Nhiệt độ cao (>30)"))
            
            # Nếu CỰC NÓNG (> 33 độ) -> Bật thêm Quạt Hút (Fan 2)
            if env.temperature > CONFIG.TEMP_EXTREME_LIMIT:
                if self.device_states[DeviceType.FAN_EXHAUST] == DeviceAction.OFF:
                    commands.append(CommandSignal(DeviceType.FAN_EXHAUST, DeviceAction.ON, "Nhiệt độ cực cao (>33)"))

        # TRƯỜNG HỢP: QUÁ LẠNH (< 20 độ)
        elif env.temperature < CONFIG.TEMP_COLD_LIMIT:
            # Bật Sưởi
            if self.device_states[DeviceType.HEATER] == DeviceAction.OFF:
                commands.append(CommandSignal(DeviceType.HEATER, DeviceAction.ON, "Nhiệt độ thấp (<20)"))
            # Tắt quạt mát nếu đang bật
            if self.device_states[DeviceType.FAN_COOLING] == DeviceAction.ON:
                commands.append(CommandSignal(DeviceType.FAN_COOLING, DeviceAction.OFF, "Tránh gió lạnh"))

        # TRƯỜNG HỢP: ỔN ĐỊNH (22 - 28 độ) -> Tắt các thiết bị làm mát/sưởi để tiết kiệm điện
        elif CONFIG.TEMP_IDEAL_MIN <= env.temperature <= CONFIG.TEMP_IDEAL_MAX:
            if self.device_states[DeviceType.FAN_COOLING] == DeviceAction.ON:
                commands.append(CommandSignal(DeviceType.FAN_COOLING, DeviceAction.OFF, "Nhiệt độ lý tưởng"))
            if self.device_states[DeviceType.FAN_EXHAUST] == DeviceAction.ON:
                commands.append(CommandSignal(DeviceType.FAN_EXHAUST, DeviceAction.OFF, "Nhiệt độ lý tưởng"))
            if self.device_states[DeviceType.HEATER] == DeviceAction.ON:
                commands.append(CommandSignal(DeviceType.HEATER, DeviceAction.OFF, "Nhiệt độ lý tưởng"))

        # --- LOGIC 2: QUẢN LÝ ĐỘ ẨM ---
        
        # KHÔ (< 50%) -> Bật Phun sương
        if env.humidity < CONFIG.HUM_DRY_LIMIT:
            if self.device_states[DeviceType.MIST] == DeviceAction.OFF:
                commands.append(CommandSignal(DeviceType.MIST, DeviceAction.ON, "Độ ẩm thấp"))
        
        # ẨM CAO (> 80%) -> Tắt Phun sương
        elif env.humidity > CONFIG.HUM_WET_LIMIT:
            if self.device_states[DeviceType.MIST] == DeviceAction.ON:
                commands.append(CommandSignal(DeviceType.MIST, DeviceAction.OFF, "Độ ẩm cao"))

        # Cập nhật trạng thái và thời gian
        for cmd in commands:
            self.device_states[cmd.device] = cmd.action
        
        if commands:
            self.last_run = datetime.now()
            
        return commands

    def update_state_manual(self, device: DeviceType, action: DeviceAction):
        """Cập nhật trạng thái khi người dùng điều khiển thủ công."""
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
        """
        Luồng chạy ngầm: Liên tục hỏi ESP32 về nhiệt độ/độ ẩm.
        """
        LogSystem.info("Khởi động dịch vụ giám sát cảm biến...")
        
        async with aiohttp.ClientSession() as session:
            while self.is_running:
                start_time = time.time()
                try:
                    # 1. Poll dữ liệu
                    async with session.get(f"{CONFIG.ESP32_BASE_URL}/status", timeout=CONFIG.POLLING_INTERVAL) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            
                            # Cập nhật trạng thái môi trường
                            self.current_env.temperature = float(data.get('temp', 0))
                            self.current_env.humidity = float(data.get('hum', 0))
                            self.current_env.timestamp = datetime.now()
                            
                            # Log nhẹ (Debug)
                            # print(f"Sensor: {self.current_env.temperature}°C | {self.current_env.humidity}%")

                except Exception:
                    # Nếu lỗi (mất kết nối), giữ nguyên giá trị cũ hoặc cảnh báo
                    # LogSystem.warning("Mất kết nối với cảm biến ESP32. Đang thử lại...")
                    pass

                # 2. Chạy logic tự động hóa (Automation)
                trend = self.brain.predict_trend(self.current_env.temperature, self.current_env.humidity)
                auto_commands = self.controller.process(self.current_env, trend)
                
                # 3. Thực thi các lệnh tự động
                for cmd in auto_commands:
                    await self.execute_device_command(cmd)
                    # Thông báo chat
                    await self.sio.emit('ai_chat_reply', {
                        'reply': f"🤖 Tự động: Tôi vừa {cmd.action.value} {cmd.device.value} vì {cmd.reason}."
                    })

                # Ngủ cho đến chu kỳ tiếp theo
                elapsed = time.time() - start_time
                sleep_time = max(0, CONFIG.POLLING_INTERVAL - elapsed)
                await asyncio.sleep(sleep_time)

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