import os
from vosk import Model, KaldiRecognizer
import wave

model_path = "models/vosk-model-vn-0.4"  # Thay bằng tên thư mục model thực tế của em
if not os.path.exists(model_path):
    print(f"❌ Không tìm thấy thư mục model tại: {model_path}")
else:
    print("✅ Đã tìm thấy thư mục model Vosk!")
    model = Model(model_path)
    print("✅ Khởi tạo Model Vosk thành công! Hệ thống Speech sẵn sàng.")