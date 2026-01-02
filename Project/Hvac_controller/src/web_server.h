#ifndef WEB_SERVER_H
#define WEB_SERVER_H

#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <SPIFFS.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "actuator_ctrl.h" 

// Liên kết biến toàn cục
extern float temperature;
extern float humidity;

const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eco Smart Monitor</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>

    <style>
        /* --- MÀU SẮC CHỦ ĐẠO --- */
        :root { 
            --primary: #d2a6ff;      
            --secondary: #d2a6ff;    
            --accent: #00d2ff;       
            --glass: rgba(0, 0, 0, 0.5); 
            --text: #fff; 
        }

        body {
            font-family: 'Roboto', sans-serif;
            background-image: url('https://wallpaper.dog/large/20492370.jpg'); 
            background-size: cover; background-attachment: fixed; background-position: center;
            color: var(--text); margin: 0; min-height: 100vh;
            display: flex; justify-content: center; align-items: flex-start;
            padding-top: 20px;
        }
        
        .overlay { position: absolute; top:0; left:0; width:100%; height:100%; background: rgba(0,0,0,0.4); z-index:-1; }
        
        .container {
            width: 95%; max-width: 600px; 
            background: rgba(20, 30, 20, 0.75);
            backdrop-filter: blur(15px); -webkit-backdrop-filter: blur(15px);
            border-radius: 20px; border: 1px solid rgba(255,255,255,0.15);
            padding: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            margin-bottom: 80px;
        }
        
        h1 { text-align: center; margin: 0; font-weight: 300; letter-spacing: 2px; text-transform: uppercase; }
        .live-indicator { width: 10px; height: 10px; background: var(--primary); border-radius: 50%; display: inline-block; box-shadow: 0 0 10px var(--primary); animation: pulse 2s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }

        /* SENSOR CARDS */
        .sensor-row { display: flex; gap: 10px; margin-top: 20px; }
        .sensor-card {
            flex: 1; background: rgba(255,255,255,0.08); padding: 15px;
            border-radius: 15px; text-align: center; border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.2s;
        }
        .sensor-card:hover { transform: translateY(-3px); border-color: var(--primary); }
        .value { font-size: 2rem; font-weight: 700; margin: 5px 0; }
        
        /* AI CARD */
        .ai-card {
            margin-top: 20px; padding: 15px; border-radius: 15px;
            background: linear-gradient(135deg, rgba(78, 255, 145, 0.15), rgba(0, 210, 255, 0.1));
            border-left: 4px solid var(--primary);
            display: flex; align-items: center; gap: 15px;
        }
        .ai-icon { font-size: 2rem; color: var(--primary); }
        .ai-text h3 { margin: 0 0 5px 0; color: var(--primary); font-size: 1rem; text-transform: uppercase; }
        .ai-text p { margin: 0; font-size: 0.9rem; opacity: 0.9; line-height: 1.4; }

        /* CHART CONTROLS */
        .chart-header { display: flex; justify-content: space-between; align-items: center; margin-top: 25px; margin-bottom: 10px; }
        .chart-controls { display: flex; gap: 8px; }
        
        .toggle-btn { 
            background: transparent; border: 1px solid rgba(255,255,255,0.3); color: white; 
            padding: 5px 12px; border-radius: 20px; cursor: pointer; font-size: 0.8rem; 
            display: flex; align-items: center; gap: 5px; transition: 0.3s;
        }
        .toggle-btn:hover { background: rgba(255,255,255,0.1); border-color: var(--primary); }
        .toggle-btn.active { background: var(--primary); color: #004d26; border-color: var(--primary); font-weight: bold; }

        .chart-container { background: rgba(0,0,0,0.3); border-radius: 15px; padding: 15px; min-height: 250px; position: relative;}

        /* GAUGE STYLES */
        .gauge-wrapper { display: none; justify-content: space-around; align-items: center; height: 100%; padding: 10px 0; }
        .gauge-box { position: relative; width: 45%; max-width: 180px; text-align: center; }
        .gauge-center-text {
            position: absolute; top: 60%; left: 50%; transform: translate(-50%, -50%);
            text-align: center; width: 100%;
        }
        .gauge-val { font-size: 1.5rem; font-weight: bold; display: block; }
        .gauge-label { font-size: 0.8rem; opacity: 0.7; }

        /* DEVICE CONTROLS */
        .control-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 20px; }
        .control-btn {
            background: rgba(255,255,255,0.05); border: none; padding: 15px; border-radius: 12px; color: white;
            display: flex; align-items: center; gap: 10px; cursor: pointer; transition: 0.3s;
        }
        .control-btn:hover { background: rgba(255,255,255,0.15); }
        .control-btn.active { background: var(--secondary); color: white; font-weight: bold; box-shadow: 0 0 15px rgba(46, 204, 113, 0.4); }
        .control-btn.active i { color: white; }

        .music-control-btn {
            position: fixed; bottom: 20px; left: 20px;
            width: 50px; height: 50px;
            background: rgba(255,255,255,0.1); border-radius: 50%;
            display: flex; justify-content: center; align-items: center;
            font-size: 1.2rem; color: var(--primary); cursor: pointer;
            border: 1px solid var(--primary); z-index: 100;
            transition: all 0.3s;
        }
        .music-control-btn:hover { background: var(--primary); color: #004d26; }

        /* DATASET SECTION */
        .dataset-section {
            margin-top: 20px; background: rgba(255,255,255,0.05);
            border-radius: 15px; padding: 20px;
            border: 1px dashed rgba(210, 166, 255, 0.5);
            text-align: center;
        }
        .upload-area { display: flex; flex-direction: column; align-items: center; gap: 10px; }
        input[type="file"] { display: none; }
        .custom-file-upload {
            border: 1px solid var(--primary); display: inline-block; padding: 10px 20px; cursor: pointer;
            border-radius: 8px; color: var(--primary); transition: 0.3s;
        }
        .custom-file-upload:hover { background: rgba(210, 166, 255, 0.2); }
        .btn-upload {
            background: var(--secondary); color: white; border: none; padding: 10px 25px;
            border-radius: 8px; cursor: pointer; font-weight: bold; margin-top: 10px; transition: 0.3s;
        }
        .btn-upload:hover { transform: scale(1.05); box-shadow: 0 0 15px var(--secondary); }
        #upload-status { margin-top: 10px; font-size: 0.9rem; min-height: 20px; }

        /* CHATBOT */
        .chat-widget { position: fixed; bottom: 20px; right: 20px; z-index: 1000; display: flex; flex-direction: column; align-items: flex-end; }
        .chat-toggle-btn {
            width: 60px; height: 60px; border-radius: 50%; background: var(--primary); color: #2e004d;
            border: none; cursor: pointer; font-size: 1.5rem; box-shadow: 0 5px 20px rgba(210, 166, 255, 0.6);
            display: flex; justify-content: center; align-items: center; transition: transform 0.3s;
        }
        .chat-toggle-btn:hover { transform: scale(1.1); }
        .chat-window {
            width: 320px; height: 450px; background: #1a1a2e; border: 1px solid var(--primary); border-radius: 20px;
            margin-bottom: 15px; display: none; flex-direction: column; overflow: hidden; box-shadow: 0 -5px 30px rgba(0,0,0,0.8);
        }
        .chat-header { background: var(--secondary); padding: 15px; color: white; font-weight: bold; display: flex; justify-content: space-between; align-items: center; }
        .chat-body { flex: 1; padding: 15px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
        .msg { padding: 8px 12px; border-radius: 12px; max-width: 80%; font-size: 0.9rem; line-height: 1.4; }
        .msg-ai { align-self: flex-start; background: rgba(255,255,255,0.1); color: #ddd; border-bottom-left-radius: 0; }
        .msg-user { align-self: flex-end; background: var(--primary); color: #2e004d; border-bottom-right-radius: 0; }
        .chat-input-area { padding: 10px; background: rgba(0,0,0,0.3); display: flex; gap: 5px; }
        .chat-input { flex: 1; padding: 10px; border-radius: 20px; border: none; outline: none; background: rgba(255,255,255,0.1); color: white; }

    </style>
</head>
<body>
    <div class="overlay"></div>
    <audio id="bgMusic" loop hidden>
        <source src="https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3" type="audio/mpeg">
    </audio>
    <div class="music-control-btn" onclick="toggleMusic()" title="Bật/Tắt nhạc nền">
        <i id="musicIcon" class="fas fa-volume-mute"></i>
    </div>
    
    <div class="container">
        <h1><span class="live-indicator"></span> ECO MONITOR</h1>
        
        <div class="sensor-row">
            <div class="sensor-card">
                <i class="fas fa-temperature-high" style="color:#ff6b6b"></i>
                <div class="value"><span id="temp">--</span>°C</div>
                <div class="unit">Nhiệt độ</div>
            </div>
            <div class="sensor-card">
                <i class="fas fa-tint" style="color:#4eff91"></i>
                <div class="value"><span id="hum">--</span>%</div>
                <div class="unit">Độ ẩm</div>
            </div>
        </div>

        <div class="dataset-section">
            <h3><i class="fas fa-brain"></i> Dạy AI (Upload Dataset)</h3>
            <div class="upload-area">
                <label for="file-upload" class="custom-file-upload">
                    <i class="fas fa-folder-open"></i> Chọn File CSV/Excel
                </label>
                <input id="file-upload" type="file" accept=".csv,.json,.txt" onchange="updateFileName()"/>
                <span id="file-name" style="color:#aaa; font-style:italic">Chưa chọn file</span>
                
                <button class="btn-upload" onclick="uploadDataset()">
                    <i class="fas fa-cloud-upload-alt"></i> Gửi cho AI học
                </button>
            </div>
            <div id="upload-status"></div>
        </div>

        <div class="ai-card">
            <div class="ai-icon"><i class="fas fa-leaf"></i></div>
            <div class="ai-text">
                <h3>Trợ lý Môi trường</h3>
                <p id="ai-advice">Đang phân tích không khí...</p>
            </div>
        </div>

        <div class="chart-header">
            <span>Biến động môi trường</span>
            <div class="chart-controls">
                <button class="toggle-btn" id="btn-switch-chart" onclick="switchChartType()">
                    <i class="fas fa-chart-pie"></i> Đổi dạng
                </button>
                <button class="toggle-btn" id="btn-chart-toggle" onclick="toggleChartSection()">
                    <i class="fas fa-eye-slash"></i> Ẩn
                </button>
            </div>
        </div>

        <div class="chart-container" id="main-chart-area">
            <canvas id="envChart"></canvas>
            <div class="gauge-wrapper" id="gauge-view">
                <div class="gauge-box">
                    <canvas id="tempGaugeCanvas"></canvas>
                    <div class="gauge-center-text">
                        <span class="gauge-val" id="gauge-temp-val" style="color:#ff6b6b">--</span>
                        <span class="gauge-label">Nhiệt độ (°C)</span>
                    </div>
                </div>
                <div class="gauge-box">
                    <canvas id="humGaugeCanvas"></canvas>
                    <div class="gauge-center-text">
                        <span class="gauge-val" id="gauge-hum-val" style="color:#4eff91">--</span>
                        <span class="gauge-label">Độ ẩm (%)</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="control-grid">
            <button class="control-btn" id="btn-fan1" onclick="toggleDevice('fan1')"><i class="fas fa-wind"></i> Quạt Mát</button>
            <button class="control-btn" id="btn-mist" onclick="toggleDevice('mist')"><i class="fas fa-tint"></i> Phun Sương</button>
            <button class="control-btn" id="btn-heater" onclick="toggleDevice('heater')"><i class="fas fa-fire"></i> Sưởi</button>
            <button class="control-btn" id="btn-fan2" onclick="toggleDevice('fan2')"><i class="fas fa-fan"></i> Quạt Hút</button>
        </div>
    </div>

    <div class="chat-widget">
        <div class="chat-window" id="chat-window">
           <div class="chat-header">
                <span><i class="fas fa-robot"></i> AI Assistant</span> <i class="fas fa-times" style="cursor:pointer" onclick="toggleChat()"></i>
            </div>
            <div class="chat-body" id="chat-body">
                <div class="msg msg-ai">Chào bạn! Tôi đang giám sát dữ liệu...</div>
            </div>
            <div class="chat-input-area">
                <input type="text" id="chat-input" class="chat-input" placeholder="Nhập lệnh (vd: bật quạt)..." onkeypress="handleEnter(event)">
                <button onclick="sendChat()" style="border:none;background:none;color:var(--primary);cursor:pointer"><i class="fas fa-paper-plane"></i></button>
            </div>
        </div>
        <button class="chat-toggle-btn" onclick="toggleChat()"><i class="fas fa-comment-dots"></i></button>
    </div>

    <script>
        // --- 1. BIẾN TOÀN CỤC ---
        var currentTemp = 0;
        var currentHum = 0;
        var isLineChart = true; 
        var lastTemp = null;
        var lastHum = null;

        // [QUAN TRỌNG] Đã thêm biến deviceStates để lưu trạng thái nút bấm
        const deviceStates = { fan1: false, fan2: false, mist: false, heater: false };

        // --- CẤU HÌNH KẾT NỐI AI ---
        // HÃY SỬA IP NÀY THÀNH IP MÁY TÍNH CỦA BẠN
        const AI_SERVER_URL = "http://192.168.1.6:3000"; 
        
        let socket = null;
        try {
            socket = io(AI_SERVER_URL);
            
            socket.on('connect', () => {
                addChatMsg("Đã kết nối thành công với bộ não AI!", 'ai');
            });
            
            socket.on('ai_chat_reply', (data) => {
                addChatMsg(data.reply, 'ai');
            });

            socket.on('update_dataset_row', (data) => {
                addChatMsg("AI đã học xong file: " + data.filename, 'ai');
            });

            socket.on('ai_command', (cmd) => {
                // Nhận lệnh từ AI và cập nhật giao diện
                console.log("Nhận lệnh AI:", cmd);
                // Cập nhật trạng thái trong biến và giao diện (không gửi lại lệnh cho ESP32 để tránh vòng lặp)
                updateButtonUI(cmd.device, cmd.action === 'on');
            });

        } catch(e) {
            console.log("Socket Error: ", e);
        }

        // --- 2. BIỂU ĐỒ ---
        var ctxLine = document.getElementById('envChart').getContext('2d');
        var lineChart = new Chart(ctxLine, {
            type: 'line',
            data: { 
                labels: [], 
                datasets: [
                    { label: 'Δ Temp', borderColor: '#ff6b6b', backgroundColor: 'rgba(255, 107, 107, 0.1)', data: [], tension: 0.4, fill: true }, 
                    { label: 'Δ Hum', borderColor: '#4eff91', backgroundColor: 'rgba(78, 255, 145, 0.1)', data: [], tension: 0.4, fill: true }
                ] 
            },
            options: { 
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: true, labels: { color: 'white' } } }, 
                scales: { x: { display: false }, y: { suggestedMin: -1, suggestedMax: 1, ticks: { color: 'rgba(255,255,255,0.7)' }, grid: { color: 'rgba(255,255,255,0.1)' } } } 
            }
        });

        const gaugeOptions = { responsive: true, maintainAspectRatio: false, cutout: '80%', rotation: -90, circumference: 180, plugins: { legend: { display: false }, tooltip: { enabled: false } } };
        var tempGauge = new Chart(document.getElementById('tempGaugeCanvas').getContext('2d'), { type: 'doughnut', data: { datasets: [{ data: [0, 50], backgroundColor: ['#555', '#222'], borderWidth: 0 }] }, options: gaugeOptions });
        var humGauge = new Chart(document.getElementById('humGaugeCanvas').getContext('2d'), { type: 'doughnut', data: { datasets: [{ data: [0, 100], backgroundColor: ['#555', '#222'], borderWidth: 0 }] }, options: gaugeOptions });

        function getTempColor(t) { return `rgb(${Math.round((t-20)/15*255)}, ${Math.round(191+(69-191)*((t-20)/15))}, ${Math.round(255+(58-255)*((t-20)/15))})`; }
        function getHumColor(h) { return `rgba(46, 204, 113, ${h/100 + 0.2})`; }

        // --- 3. CẬP NHẬT DỮ LIỆU ---
        function updateData() {
            fetch('/status').then(r => r.json()).then(d => {
                currentTemp = d.temp; currentHum = d.hum;
                document.getElementById('temp').innerText = d.temp;
                document.getElementById('hum').innerText = d.hum;

                let tStr = new Date().toLocaleTimeString();
                let deltaTemp = (lastTemp !== null) ? parseFloat((d.temp - lastTemp).toFixed(2)) : 0;
                let deltaHum = (lastHum !== null) ? parseFloat((d.hum - lastHum).toFixed(2)) : 0;
                lastTemp = d.temp; lastHum = d.hum;

                lineChart.data.labels.push(tStr);
                lineChart.data.datasets[0].data.push(deltaTemp);
                lineChart.data.datasets[1].data.push(deltaHum);
                if(lineChart.data.labels.length > 20) { lineChart.data.labels.shift(); lineChart.data.datasets[0].data.shift(); lineChart.data.datasets[1].data.shift(); }
                if(isLineChart) lineChart.update();

                // Gauge Update
                let tVal = d.temp > 50 ? 50 : d.temp;
                tempGauge.data.datasets[0].data = [tVal, 50 - tVal];
                tempGauge.data.datasets[0].backgroundColor = [getTempColor(d.temp), 'rgba(255,255,255,0.05)'];
                document.getElementById('gauge-temp-val').innerText = d.temp;
                document.getElementById('gauge-temp-val').style.color = getTempColor(d.temp);
                if(!isLineChart) tempGauge.update();

                let hVal = d.hum;
                humGauge.data.datasets[0].data = [hVal, 100 - hVal];
                humGauge.data.datasets[0].backgroundColor = [getHumColor(d.hum), 'rgba(255,255,255,0.05)'];
                document.getElementById('gauge-hum-val').innerText = d.hum;
                if(!isLineChart) humGauge.update();

                analyzeEnvironment(d.temp, d.hum);
            }).catch(err => console.log("ESP32 Offline"));
        }

        // --- 4. HÀM GIAO DIỆN ---
        function switchChartType() {
            isLineChart = !isLineChart;
            document.getElementById('envChart').style.display = isLineChart ? 'block' : 'none';
            document.getElementById('gauge-view').style.display = isLineChart ? 'none' : 'flex';
            document.getElementById('btn-switch-chart').innerHTML = isLineChart ? '<i class="fas fa-chart-pie"></i> Đổi dạng' : '<i class="fas fa-chart-line"></i> Đổi dạng';
            if(!isLineChart) { tempGauge.update(); humGauge.update(); }
        }
        function toggleChartSection() {
            let d = document.getElementById("main-chart-area");
            d.style.display = d.style.display === "none" ? "block" : "none";
        }
        function toggleChat() {
            let w = document.getElementById("chat-window");
            w.style.display = w.style.display === "flex" ? "none" : "flex";
        }

        // --- 5. HÀM ĐIỀU KHIỂN THIẾT BỊ (ĐÃ SỬA LỖI) ---
        function toggleDevice(device) {
            // Đảo trạng thái logic
            const newState = !deviceStates[device];
            const stateStr = newState ? "on" : "off";

            // Gửi lệnh xuống ESP32
            fetch(`/control?device=${device}&state=${stateStr}`)
                .then(res => {
                    if (res.ok) {
                        updateButtonUI(device, newState);
                    }
                });
        }
        
        function updateButtonUI(device, isOn) {
            deviceStates[device] = isOn;
            const btn = document.getElementById(`btn-${device}`);
            if (btn) {
                if (isOn) {
                    btn.classList.remove('active'); // Xóa class cũ nếu có style khác
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            }
        }

        // --- 6. HÀM KHÁC ---
        function analyzeEnvironment(t, h) {
            let msg = "";
            if (t == 0) return; // Chưa có dữ liệu

            if (t > 32) msg = "⚠️ <b>Nắng nóng:</b> Nhiệt độ cao ảnh hưởng đến cây trồng và sức khỏe.";
            else if (t > 28) msg = "Hơi nóng. Giữ không gian thoáng mát để tiết kiệm năng lượng.";
            else if (t < 20) msg = "❄️ <b>Se lạnh:</b> Nhiệt độ thấp, thích hợp bảo quản thực phẩm.";
            else msg = "🌱 <b>Tuyệt vời:</b> Môi trường sinh thái lý tưởng.";

            if (h > 80) msg += "<br>💧 <b>Ẩm ướt:</b> Độ ẩm rừng nhiệt đới, cẩn thận nấm mốc.";
            else if (h < 40) msg += "<br>🌵 <b>Khô hạn:</b> Không khí quá khô, cần bổ sung độ ẩm.";
            
            document.getElementById("ai-advice").innerHTML = msg;
        }

        function updateFileName() {
            let input = document.getElementById('file-upload');
            document.getElementById('file-name').innerText = input.files.length > 0 ? input.files[0].name : "Chưa chọn file";
        }

        function uploadDataset() {
            let input = document.getElementById('file-upload');
            if(input.files.length === 0) return alert("Chưa chọn file!");
            let reader = new FileReader();
            reader.onload = function(e) {
                if(socket) socket.emit('upload_dataset', { filename: input.files[0].name, content: e.target.result });
            };
            reader.readAsText(input.files[0]);
        }

        function sendChat() {
            let input = document.getElementById('chat-input');
            let val = input.value.trim();
            if(val && socket) {
                addChatMsg(val, 'user');
                socket.emit('user_chat_message', { message: val });
                input.value = '';
            }
        }
        function handleEnter(e) { if(e.key==='Enter') sendChat(); }
        
        function addChatMsg(text, type) {
            let div = document.createElement('div');
            div.className = `msg msg-${type}`;
            div.innerText = text;
            let body = document.getElementById('chat-body');
            body.appendChild(div);
            body.scrollTop = body.scrollHeight;
        }

        // Music Control
        var bgMusic = document.getElementById("bgMusic");
        var isMusicPlaying = false;
        function toggleMusic() {
            if (isMusicPlaying) { bgMusic.pause(); document.getElementById("musicIcon").className = "fas fa-volume-mute"; }
            else { bgMusic.play(); document.getElementById("musicIcon").className = "fas fa-volume-up"; }
            isMusicPlaying = !isMusicPlaying;
        }

        window.onload = function() {
            setInterval(updateData, 2000);
        };
    </script>
</body>
</html>
)rawliteral";

AsyncWebServer server(80);

void setupWebServer() {
    if(WiFi.status() == WL_CONNECTED) {
        Serial.print("[WebServer] IP: ");
        Serial.println(WiFi.localIP());
    }

    server.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
        request->send_P(200, "text/html", index_html);
    });

    server.on("/control", HTTP_GET, [](AsyncWebServerRequest *request){
         if (request->hasParam("device") && request->hasParam("state")) {
            String device = request->getParam("device")->value();
            String state = request->getParam("state")->value();
            bool turnOn = (state == "on");
            if(device == "fan1") controlFan1(turnOn);
            else if(device == "mist") controlMist(turnOn);
            else if(device == "heater") controlHeater(turnOn);
            else if(device == "fan2") controlFan2(turnOn);
            request->send(200, "text/plain", "OK");
         } else request->send(400);
    });

    server.on("/status", HTTP_GET, [](AsyncWebServerRequest *request){
        String json = "{";
        json += "\"temp\":" + String(temperature, 1) + ",";
        json += "\"hum\":" + String(humidity, 1);
        json += "}";
        request->send(200, "application/json", json);
    });

    server.begin();
    Serial.println("[WebServer] Da khoi dong!");
}

#endif