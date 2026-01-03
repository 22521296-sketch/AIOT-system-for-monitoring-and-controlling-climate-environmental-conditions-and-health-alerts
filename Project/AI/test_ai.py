import unittest
from datetime import datetime, timedelta # <--- Thêm thư viện này
from ai_agent import AutomationController, EcoBrain, SensorData, DeviceType, DeviceAction

class TestEcoSystem(unittest.TestCase):
    def setUp(self):
        self.controller = AutomationController()
        self.brain = EcoBrain()
        
        # --- SỬA LỖI TẠI ĐÂY ---
        # Thay vì gán = 0, ta gán bằng thời điểm cách đây 1 giờ
        # Để đảm bảo (datetime.now() - last_run) sẽ ra kết quả hợp lệ
        self.controller.last_run = datetime.now() - timedelta(hours=1)

    def test_tc01_ideal(self):
        """TC01: Môi trường lý tưởng -> Tắt hết"""
        print("\n--- Running TC01 ---")
        # Reset thời gian chờ
        self.controller.last_run = datetime.now() - timedelta(hours=1)
        
        env = SensorData(temperature=25, humidity=60)
        
        cmds = self.controller.process(env, self.brain)
        
        # Kiểm tra không có lệnh bật nào
        found_on_cmd = False
        for cmd in cmds:
            if cmd.action == DeviceAction.ON:
                found_on_cmd = True
                break
        self.assertFalse(found_on_cmd, "Lỗi: Đáng lẽ phải tắt hết thiết bị nhưng lại có thiết bị Bật")
            
    def test_tc02_hot_threshold(self):
        """TC02: Vừa chớm nóng 30.1 -> Bật Fan 1"""
        print("\n--- Running TC02 ---")
        self.controller.last_run = datetime.now() - timedelta(hours=1)
        
        env = SensorData(temperature=30.1, humidity=60) # > 30.0
        
        cmds = self.controller.process(env, self.brain)
        
        # Tìm lệnh bật Fan 1
        fan1_cmd = next((c for c in cmds if c.device == DeviceType.FAN_COOLING), None)
        
        if fan1_cmd is None:
            self.fail("Lỗi: Không bật quạt dù nhiệt độ 30.1 > 30.0")
            
        self.assertEqual(fan1_cmd.action, DeviceAction.ON)
        print(f"  [PASS] Lý do bật: {fan1_cmd.reason}")

    def test_tc08_ai_prediction(self):
        """TC08: AI dự báo nóng (Pre-emptive) -> Bật Fan sớm"""
        print("\n--- Running TC08 (AI Predictive) ---")
        self.controller.last_run = datetime.now() - timedelta(hours=1)
        
        # Hiện tại 28 độ (Mát)
        env = SensorData(temperature=28.0, humidity=60)
        
        # Giả lập AI dự đoán tương lai sẽ lên 31 độ
        # Lưu ý: Cần gán đúng kiểu dữ liệu trả về của hàm predict_next
        original_predict = self.brain.predict_next
        self.brain.predict_next = lambda t, h: (31.0, "Tăng mạnh") 
        
        cmds = self.controller.process(env, self.brain)
        
        # Khôi phục hàm cũ
        self.brain.predict_next = original_predict
        
        # Kỳ vọng: Phải bật quạt
        fan1_cmd = next((c for c in cmds if c.device == DeviceType.FAN_COOLING), None)
        
        if fan1_cmd:
            print(f"  [PASS] AI hoạt động đúng: {fan1_cmd.reason}")
            self.assertEqual(fan1_cmd.action, DeviceAction.ON)
        else:
            self.fail("AI thất bại: Không bật quạt đón đầu dù dự báo nóng!")

if __name__ == '__main__':
    unittest.main()
