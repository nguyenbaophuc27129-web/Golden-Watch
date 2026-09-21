# -*- coding: utf-8 -*-
"""NK-33: DÒ YOLO-POSE TRỰC TIẾP TRÊN CAMERA (không cần server).

Chụp 5 khung cách 0.5s từ webcam 720p → chạy YOLOv8n-pose conf 0.05 ở 2 mức
imgsz (640 / 960) trên khung cuối → in từng detection (box, conf, số keypoint
tin cậy) + độ sáng khung + LƯU ẢNH ĐÃ VẼ vào exports/ để nhìn đúng camera
thấy gì. Dùng để chẩn đoán: người ngồi sát/nửa thân bị bỏ vì sao.

Chạy:  python src/probe_yolo_frame.py
"""
import os
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
for p in (SRC_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from ultralytics import YOLO   # noqa: E402

MODEL_PATH = os.path.join(SRC_DIR, 'yolov8n-pose.pt')
OUT_DIR = os.path.join(PROJECT_ROOT, 'exports')
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]


def annotate(frame, res, imgsz):
    h, w = frame.shape[:2]
    kpts = res[0].keypoints
    boxes = res[0].boxes
    n_det = 0 if boxes is None else len(boxes)
    if kpts is not None and kpts.xy is not None:
        for person in kpts.xy:
            pts = [(int(x * w), int(y * h))
                   for x, y in person.cpu().numpy()]
            for a, b in SKELETON:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], (80, 220, 80), 3)
            for p in pts:
                cv2.circle(frame, p, 4, (0, 255, 255), -1)
    if boxes is not None:
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            c = float(b.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(frame, f'{c:.2f}', (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    cv2.putText(frame, f'imgsz {imgsz} — det {n_det}',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame, n_det


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print('KHONG MO DUOC CAMERA')
        return
    # NK-34: warm-up — bỏ khung đen auto-exposure (t0 ~5/255 → t+6s ổn),
    # nếu không YOLO "bịa người" trên khung đen làm sai chẩn đoán.
    t0 = time.time()
    while time.time() - t0 < 6.0:
        ok, fr = cap.read()
        if ok and fr is not None and \
                cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).mean() >= 20:
            print(f'Camera sang sau {time.time() - t0:.1f}s')
            break
        time.sleep(0.1)
    print('Chup 5 khung cach 0.5s — NGỒI YÊN tại vị trí đang test...')
    frame = None
    for i in range(5):
        ok, frame = cap.read()
        time.sleep(0.5)
    cap.release()
    if not ok or frame is None:
        print('KHONG DOC DUOC KHUNG')
        return
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f'Khung: {frame.shape[1]}x{frame.shape[0]}  '
          f'do sang TB: {gray.mean():.0f}/255')

    model = YOLO(MODEL_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)
    for imgsz in (640, 960):
        res = model.predict(frame, verbose=False, conf=0.05, imgsz=imgsz)
        boxes = res[0].boxes
        kpts = res[0].keypoints
        n_det = 0 if boxes is None else len(boxes)
        print(f'--- imgsz {imgsz}: {n_det} detection (conf>=0.05)')
        if boxes is not None:
            for j, b in enumerate(boxes):
                conf = float(b.conf[0])
                cls = int(b.cls[0])
                n_kp = 0
                if (kpts is not None and kpts.conf is not None
                        and j < len(kpts.conf)):
                    n_kp = int((kpts.conf[j].cpu().numpy() > 0.3).sum())
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                print(f'    #{j} cls={cls} conf={conf:.3f} '
                      f'box=({x1},{y1})-({x2},{y2}) kp_conf>0.3: {n_kp}/17')
        out = os.path.join(OUT_DIR, f'probe_yolo_i{imgsz}.jpg')
        cv2.imwrite(out, annotate(frame.copy(), res, imgsz))
        print(f'    anh: {out}')
    print('XONG — so sánh 2 ảnh exports/probe_yolo_i640.jpg / i960.jpg')


if __name__ == '__main__':
    main()
