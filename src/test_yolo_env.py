from ultralytics import YOLO
import cv2

print("🚀 Đang tải mô hình YOLOv8n-pose...")
model = YOLO('yolov8n-pose.pt')

print("✅ Đang bật Webcam test khung xương dáng đi (Nhấn 'q' để thoát)...")
results = model(source=0, show=True, conf=0.5, stream=True)

for r in results:
    # Stream chạy liên tục cho đến khi bấm phím q trên cửa sổ hiển thị
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()