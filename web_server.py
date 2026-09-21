# -*- coding: utf-8 -*-
"""
GOLDEN-WATCH — WEB SERVER GIÁM SÁT THỜI GIAN THỰC (FastAPI, chạy LOCAL)

Lý do thay Streamlit cho video: Streamlit rerun cả script mỗi tick → video
giật; FastAPI + MJPEG stream = đúng kiểu web camera giám sát (20–30 fps).

Kiến trúc 4 luồng trong 1 process:
  Thread 1 — CAMERA   : đọc webcam 720p liên tục ~30fps, vẽ khung xương YOLO
                        từng khung + HUD méo mặt MediaPipe → frame phục vụ
  Thread 2 — RADAR    : LD2450 quét 0.6s/nhịp (không chặn camera)
  Thread 3 — SPEECH   : NK-32 ghi 5s → ML nói khó + Vosk transcript (lazy
                        init; kết quả giữ 90s — hết hạn fusion tự bỏ qua)
  Thread 4 — PHÂN TÍCH: mỗi 2s face/arm/gait (+ speech còn tươi) → Defense
                        → Fusion → Alert → NIHSS → Handoff (vào /api/state)

Trang web:  http://localhost:5001      (dashboard CCTV + cảnh báo + form)
Video:      http://localhost:5001/video.mjpg   (MJPEG — mở bằng VLC được)
API:        /api/state · /api/radar (POST) · /patient (POST)
            /alert/test (POST) · /handoff/qr · /handoff/pdf

KHÔNG PHẢN CHẨN ĐOÁN — công cụ hỗ trợ. Khẩn cấp: GỌI 115.
Chạy:  python web_server.py
"""

import os
import sys
import time
import json
import asyncio
import threading
from collections import deque

import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
for p in (SRC_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# --lan: nghe trên MẠNG CỤC BỘ (hotspot/wifi nhà) để app điện thoại kết nối.
# Mặc định KHÔNG có cờ = 127.0.0.1 (an toàn như cũ, điện thoại không thấy).
HOST = '0.0.0.0' if '--lan' in sys.argv else '127.0.0.1'

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import (HTMLResponse, JSONResponse, Response,
                               StreamingResponse, FileResponse,
                               RedirectResponse)

from fusion.fusion_engine import FusionEngine
from fusion.nihss_estimator import estimate_nihss, calculate_nihss_ci
from fusion.triage_engine import TriageEngine
from defense.defense_engine import DefenseEngine
from defense.personal_profile import PersonalProfile
from alerts.alert_system import AlertSystem
from handoff.handoff_system import HandoffSystem
from detection.radar_module import RadarModule

MODELS = os.path.join(PROJECT_ROOT, 'models')
JPEG_QUALITY = 70

# ------------------------------------------------------------------
# KIỂM TRA CỔNG TRƯỚC KHI LOAD MODEL (đỡ chờ 10s rồi mới báo lỗi)
# ------------------------------------------------------------------
import socket as _socket
with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
    _s.settimeout(0.5)
    if _s.connect_ex(('127.0.0.1', 5001)) == 0:
        print('=' * 64)
        print('  LỖI: Cổng 5001 ĐANG ĐƯỢC DÙNG — server cũ chưa tắt.')
        print('  Cách tắt bản cũ (chọn 1):')
        print('    1) Tìm cửa sổ server cũ nhấn Ctrl+C')
        print('    2) Hoặc chạy lệnh:')
        print('       netstat -ano | findstr :5001')
        print('       taskkill /F /PID <số PID ở cột cuối>')
        print('=' * 64)
        sys.exit(1)

# ======================================================================
# LOAD MODEL 1 LẦN
# ======================================================================
print('[WEB] Đang tải model (chờ ~10 giây)...')
from detection.face_module_v7 import FaceAsymmetryDetector      # noqa: E402
from detection.face_ml_v3 import FaceMLV3                       # noqa: E402
from detection.arm_module import ArmWeaknessDetector             # noqa: E402
from detection.gait_module import GaitPoseDetector               # noqa: E402

FACE = FaceAsymmetryDetector(
    model_path=os.path.join(MODELS, 'face_landmarker_v2.task'))
FACE.ml_model = FaceMLV3.load_latest(MODELS)   # SYS-29: ML v3 28 đặc trưng
print('[WEB] Face ML v3:',
      (f'ON — artifact {FACE.ml_model.created}, '
       f'AUC test {FACE.ml_model.auc_test}') if FACE.ml_model
      else 'OFF (không có artifact face_blend_v3_*)')
# NK-28: detector mặt RIÊNG cho HUD real-time (thread camera) — mặt xuất
# hiện đâu trong khung góc rộng là khóa NGAY, không chờ chu kỳ phân tích
# 2s. Instance riêng vì FaceLandmarker VIDEO mode không thread-safe;
# ML v3 là toán thuần đọc artifact nên chia sẻ được.
FACE_LIVE = FaceAsymmetryDetector(
    model_path=os.path.join(MODELS, 'face_landmarker_v2.task'))
FACE_LIVE.ml_model = FACE.ml_model
ARM = ArmWeaknessDetector(
    pose_model_path=os.path.join(SRC_DIR, 'yolov8n-pose.pt'),
    ml_model_path=os.path.join(MODELS, 'arm_weakness_20260830_200657.pth'),
    scaler_path=os.path.join(MODELS,
                             'arm_weakness_20260830_200657_scaler.pkl'))
GAIT = GaitPoseDetector(model_path=os.path.join(SRC_DIR, 'yolov8n-pose.pt'))
YOLO_LOCK = threading.Lock()

# ======================================================================
# TRẠNG THÁI DÙNG CHUNG (global — 1 phiên giám sát / server)
# ======================================================================
STATE = {
    'patient': {'name': '', 'age': '', 'gender': 'Nam', 'phone': '',
                'address': ''},
    'radar': RadarModule(simulation=True, sim_scenario='normal'),
    'radar_mode': '🟡 SIMULATION',
    'radar_history': deque(maxlen=120),
    'cam_frame': None,          # frame thô (worker camera ghi)
    'cam_annotated': None,      # frame đã vẽ YOLO+HUD (phục vụ MJPEG)
    'cam_error': None,
    'hud_face': None,           # kết quả face gần nhất (chu kỳ 2s)
    'face_live': None,          # NK-28: face real-time từ thread camera
    'arm_metrics': None,        # NK-28: metrics tay cho bảng HUD
    'modules': {},
    'fused_score': 0.0,
    'risk_level': '—',
    'verdict': {'alert': False, 'reason': '', 'trend': 'NO_DATA'},
    'nihss': None,
    'last_alert': None,         # cảnh báo gần nhất (banner đỏ)
    'motion': 0.0,
    'prev_gray': None,
    'camera_online': False,
    'analysis_count': 0,
    'speech_result': None,      # NK-32: kết quả speech gần nhất (thread 3)
    'speech_ts': 0.0,           # time.time() lúc có kết quả
    'speech_note': '',          # "im lang — dang nghe" / lỗi
    'speech_error': None,
}
CAM_LOCK = threading.Lock()
RADAR_LOCK = threading.Lock()
ANALYSIS_LOCK = threading.Lock()

_WARN_TS = {}
_WARN_LOCK = threading.Lock()


def _rate_warn(key, msg, period_s=30):
    """NK-32: in cảnh báo tối đa 1 lần/period — chẩn đoán được lỗi lặp
    (trước đây draw_live_pose nuốt exception im lặng, không phân biệt được
    'không phát hiện' với 'văng exception')."""
    now = time.time()
    with _WARN_LOCK:
        if now - _WARN_TS.get(key, 0) < period_s:
            return
        _WARN_TS[key] = now
    print(f'[WARN {key}] {msg}')

# ------------------------------------------------------------------
# MOBILE ALERT (NK-25): kênh sự kiện đẩy sang app điện thoại (PWA).
# Thread phân tích push sự kiện (status mỗi chu kỳ 2s + alert khi có);
# mỗi client WebSocket /ws đọc các sự kiện có seq lớn hơn lần đọc cuối.
# Thuần ADD-ON: không đổi pipeline, không đổi điểm, không cần internet.
# ------------------------------------------------------------------
MOBILE_LOCK = threading.Lock()
STATE['mobile_seq'] = 0
STATE['mobile_events'] = deque(maxlen=100)


def push_mobile_event(etype, **kw):
    with MOBILE_LOCK:
        STATE['mobile_seq'] += 1
        ev = {'type': etype, 'seq': STATE['mobile_seq'],
              't_server': time.strftime('%H:%M:%S'), **kw}
        STATE['mobile_events'].append(ev)
    return ev


def build_mobile_snapshot():
    with ANALYSIS_LOCK:
        snap = {k: STATE.get(k) for k in ('modules', 'fused_score',
                                          'risk_level', 'verdict', 'nihss',
                                          'last_alert', 'camera_online',
                                          'analysis_count', 'patient',
                                          'radar_mode')}
    with MOBILE_LOCK:
        snap['seq'] = STATE['mobile_seq']
    snap['type'] = 'snapshot'
    snap['t_server'] = time.strftime('%H:%M:%S')
    snap['host_lan'] = detect_lan_ip()
    return snap


def detect_lan_ip():
    """IP LAN của laptop (để điện thoại mở http://<ip>:5001/mobile).
    Kỹ thuật: UDP-connect KHÔNG gói tin nào gửi đi — chỉ tra bảng định tuyến."""
    import socket
    for probe in ('192.168.137.1', '8.8.8.8'):   # 137.1 = subnet hotspot Win
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.2)
            s.connect((probe, 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and not ip.startswith('127.'):
                return ip
        except Exception:
            continue
    return '127.0.0.1'

FUSION = FusionEngine()
DEFENSE = DefenseEngine()
ALERTER = AlertSystem(buzzer_enabled=True, cooldown_seconds=60)
TRIAGE = TriageEngine()
HANDOFF = HandoffSystem(patient_info=STATE['patient'])
# CHẾ ĐỘ HỌC 3 NGÀY: ghi đặc điểm riêng người dùng (gù/cong lung/nhech mép
# thói quen...) → đủ 3 ngày tự calibrate Defense (bật L4 cá nhân hóa)
PROFILE = PersonalProfile(os.path.join(MODELS, 'personal_profile.json'))

YOLO_SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4),
                 (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                 (5, 11), (6, 12), (11, 12),
                 (11, 13), (13, 15), (12, 14), (14, 16)]
FACE_THRESHOLDS = {'mouth_ratio': (0.25, '%'), 'eye_ratio': (0.30, '%'),
                   'face_tilt': (10.0, 'do'),
                   'nasolabial_ratio': (0.35, '%'),
                   'forehead_ratio': (0.30, '%')}
FACE_NAMES = {'mouth_ratio': 'meo mieng', 'eye_ratio': 'lech nhan',
              'face_tilt': 'nghieng dau',
              'nasolabial_ratio': 'ranh mui-moi',
              'forehead_ratio': 'nep tran'}

# NK-28: mạng landmark mặt theo ĐÚNG mẫu Google Face Landmarker
# ([mediapipe_python_tasks]_face_landmarker.py): tesselation 2556 đoạn +
# viền mặt/mắt/môi 124 đoạn + 2 vòng iris
from mediapipe.tasks.python.vision import FaceLandmarksConnections as _FLC  # noqa: E402
_FACE_TESS = np.array([(c.start, c.end)
                       for c in _FLC.FACE_LANDMARKS_TESSELATION],
                      dtype=np.int32)
_FACE_CONT = np.array([(c.start, c.end)
                       for c in _FLC.FACE_LANDMARKS_CONTOURS],
                      dtype=np.int32)
_FACE_IRIS = np.array([(c.start, c.end) for c in
                       list(_FLC.FACE_LANDMARKS_LEFT_IRIS)
                       + list(_FLC.FACE_LANDMARKS_RIGHT_IRIS)],
                      dtype=np.int32)


def draw_face_mesh(frame, lm, color):
    """Vẽ mạng landmark QUANH MẶT (kiểu file mẫu Google): lưới mảnh phủ
    mặt + viền đậm theo màu trạng thái + 2 iris vàng. 478 landmark tự
    gắn vào mặt BẤT KỲ đâu trong khung — camera góc rộng vẫn bám được."""
    h, w = frame.shape[:2]
    pts = np.column_stack([lm[:, 0] * w, lm[:, 1] * h]).astype(np.int32)
    for a, b in _FACE_TESS:
        pa, pb = pts[a], pts[b]
        cv2.line(frame, (pa[0], pa[1]), (pb[0], pb[1]),
                 (140, 150, 140), 1)
    for a, b in _FACE_CONT:
        pa, pb = pts[a], pts[b]
        cv2.line(frame, (pa[0], pa[1]), (pb[0], pb[1]),
                 color, 2, cv2.LINE_AA)
    for a, b in _FACE_IRIS:
        pa, pb = pts[a], pts[b]
        cv2.line(frame, (pa[0], pa[1]), (pb[0], pb[1]),
                 (40, 230, 230), 1, cv2.LINE_AA)
    return frame


# ======================================================================
# VẼ OVERLAY (khung xương YOLO + HUD méo mặt MediaPipe)
# ======================================================================
def _box_iou(a, b):
    """IoU 2 box xyxy (numpy) — dùng lọc persistence khung xương."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


_POSE_LAST_BOXES = []   # box người khung trước — NK-34 chống xương "chạy lung tung"


def draw_live_pose(frame, conf=0.10):
    """Khung xương YOLO vẽ TỪNG KHUNG (tối đa 3 người). NK-33: conf
    0.25→0.10 + imgsz 480→640. NK-34 thêm LỌC BỀN VỮNG: detection conf thấp
    (<0.15) chỉ vẽ khi box đó đã có ở KHUNG TRƯỚC (IoU>0.3) — khử các
    "người bịa" nhảy lung tung trên nhiễu/khung đen khởi động. CHỈ HIỂN
    THỊ — đường chấm điểm dùng detector riêng của arm/gait, KHÔNG đổi."""
    global _POSE_LAST_BOXES
    if ARM.pose_model is None:
        return frame
    try:
        with YOLO_LOCK:
            res = ARM.pose_model.predict(frame, verbose=False, conf=conf,
                                         imgsz=640)
        kpts = res[0].keypoints
        boxes = res[0].boxes
        h, w = frame.shape[:2]
        n_pts = 0
        n_people = 0
        kept_boxes = []
        if (kpts is not None and kpts.xy is not None
                and boxes is not None and len(kpts.xy) > 0):
            confs = boxes.conf.cpu().numpy()
            # NK-28: đoạn CÁNH TAY tô CYAN đậm (5-7-9 trái, 6-8-10 phải) —
            # đúng vùng đo lệch nhẹ; phần xương còn lại xanh lá
            ARM_BONES = {(5, 7), (7, 9), (6, 8), (8, 10)}
            for j, person in enumerate(kpts.xy[:3]):
                cf = float(confs[j]) if j < len(confs) else 0.0
                if cf < 0.15:
                    bx = (boxes.xyxy[j].cpu().numpy()
                          if j < len(boxes.xyxy) else None)
                    # conf thấp: chỉ nhận khi người này đã bền ở khung trước
                    if bx is None or not any(_box_iou(bx, pb) > 0.3
                                             for pb in _POSE_LAST_BOXES):
                        continue
                    bx = tuple(float(v) for v in bx)
                else:
                    bx = None
                # NK-36: LỌC THEO ĐỘ TIN CẬY TỪNG ĐIỂM (conf keypoint >= 0.3)
                # — trước đây vẽ CẢ 17 điểm kể cả điểm nhiễu (conf 0.05) có
                # tọa độ rải ngẫu nhiên → xương vẽ loạn ("chạy lung tung")
                # đúng khi đã phát hiện được người. YOLO luôn trả đủ 17 cặp
                # tọa độ nhưng điểm yếu phải BỎ, không vẽ.
                kpc = (kpts.conf[j].cpu().numpy()
                       if (kpts.conf is not None and j < len(kpts.conf))
                       else None)
                raw = person.cpu().numpy()
                pts, ok_pts = [], []
                for i, (x, y) in enumerate(raw):
                    good = kpc is None or kpc[i] >= 0.3
                    pts.append((int(x * w), int(y * h)))
                    if good:
                        ok_pts.append(i)
                ok = set(ok_pts)
                for a, b in YOLO_SKELETON:
                    if a in ok and b in ok:
                        col = (230, 200, 60) if (a, b) in ARM_BONES \
                            else (80, 220, 80)
                        cv2.line(frame, pts[a], pts[b], col, 3, cv2.LINE_AA)
                for i in ok_pts:
                    p = pts[i]
                    r = 5 if i in (5, 6, 7, 8, 9, 10) else 4
                    cv2.circle(frame, p, r, (0, 255, 255), -1, cv2.LINE_AA)
                    n_pts += 1
                n_people += 1
                if bx is not None:
                    kept_boxes.append(bx)
            kept_boxes += [tuple(float(v) for v in boxes.xyxy[j].cpu().numpy())
                           for j in range(n_people, min(len(kpts.xy), 3))
                           if j < len(boxes.xyxy)]
        _POSE_LAST_BOXES = kept_boxes
        STATE['pose_people'] = n_people
        # NK-33: nhãn LUÔN hiện (kể cả NGUOI: 0) — phân biệt được
        # "overlay chết" với "YOLO chạy đúng nhưng không thấy người"
        cv2.putText(frame,
                    f'NGUOI: {n_people}  DIEM CO THE: {n_pts}/{n_people * 17}',
                    (w - 320, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (80, 220, 80) if n_people else (160, 160, 160), 1)
    except Exception as e:
        _rate_warn('yolo_draw', f'draw_live_pose: {e!r}', 30)
    return frame


def draw_face_hud(frame, face_r):
    """HUD méo mặt kiểu GIÁM SÁT: landmark tự gắn vào mặt BẤT KỲ ở đâu
    trong khung (không ép vị trí). Camera gác cao vẫn đo được."""
    h, w = frame.shape[:2]
    status = face_r.get('status', 'NO_FACE')
    lost = status in ('NO_FACE', 'INVALID_LANDMARKS', 'PARTIAL_FACE',
                      'NO_DETECTOR', 'ERROR')
    color = {'DANGER': (0, 0, 255), 'WARNING': (0, 165, 255),
             'NORMAL': (60, 200, 60)}.get(status, (0, 0, 255))
    lm = face_r.get('raw_landmarks')
    if lm is not None:
        # NK-28: mạng landmark đầy đủ quanh mặt (thay 239 chấm rời)
        frame = draw_face_mesh(frame, lm, color)
        xs, ys = lm[:, 0], lm[:, 1]
        cv2.rectangle(frame, (int(xs.min() * w), int(ys.min() * h)),
                      (int(xs.max() * w), int(ys.max() * h)), color, 2)
    score = float(face_r.get('score', 0) or 0)
    bar_y = h - 30
    cv2.rectangle(frame, (10, bar_y), (w - 10, bar_y + 14), (50, 50, 50), -1)
    fill = int(min(max(score, 0), 100) / 100 * (w - 20))
    cv2.rectangle(frame, (10, bar_y), (10 + fill, bar_y + 14), color, -1)
    cv2.line(frame, (10 + int(0.30 * (w - 20)), bar_y - 4),
             (10 + int(0.30 * (w - 20)), bar_y + 18), (0, 165, 255), 2)
    label = f'MEO MAT {score:.0f}/100  (WARNING >= 30)'
    if face_r.get('hold'):
        label += f'  [HOLD {face_r.get("hold_age_s", 0)}s]'
    cv2.putText(frame, label, (10, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    rm = face_r.get('raw_metrics') or {}
    y = 26
    tilt = rm.get('face_tilt')
    if tilt is not None:
        cv2.putText(frame, f'nghieng dau: {tilt:.0f} do (WARNING khi ~>=25)',
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 165, 255) if tilt > 10 else (255, 255, 255), 1)
        y += 22
    yaw, pitch = rm.get('head_yaw'), rm.get('head_pitch')
    if yaw is not None and pitch is not None and y <= 100:
        cv2.putText(frame,
                    f'chuyen dau (matrix MP): quay {yaw:.0f} | nga {pitch:.0f} do',
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 255), 1)
        y += 22
    ml = rm.get('ml_prob')
    if ml is not None and y <= 100:
        cv2.putText(frame,
                    f'ML v3 (28 ft): {ml:.0f}%  |  rules: '
                    f'{rm.get("score_rules", 0):.0f}',
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (120, 255, 120), 1)
        y += 22
    for k, (t, unit) in FACE_THRESHOLDS.items():
        if k == 'face_tilt' or y > 100:
            continue
        v = rm.get(k)
        if v is not None and v / 100.0 > t:
            cv2.putText(frame,
                        f'! {FACE_NAMES[k]}: {v:.0f}{unit} > {int(t * 100)}%',
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 165, 255), 1)
            y += 22
    if lost and not face_r.get('hold'):
        cv2.putText(frame, 'MAT: KHONG THAY MAT — landmark tu gan khi co mat',
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 1)
    return frame


def draw_arm_panel(frame):
    """NK-28: bảng chi tiết TAY từ xương YOLO (góc vai-khuỷu-cổ tay,
    độ rơi, lệch L/R) + CẢNH BÁO LỆCH NHẸ ngay khi lệch góc >10 độ hoặc
    chênh cao cổ tay >30px — HIỂN THỊ THÊM, không đổi status/điểm."""
    m = STATE.get('arm_metrics') or {}
    if not m:
        return frame
    h, _w = frame.shape[:2]
    y = h - 66
    mild = bool(m.get('mild_asym'))
    col = (0, 165, 255) if mild else (255, 255, 255)
    cv2.putText(frame,
                f"TAY L: goc {m.get('left_arm_angle', 0):.0f} do | "
                f"R: {m.get('right_arm_angle', 0):.0f} do | "
                f"lech goc {m.get('angle_asymmetry', 0):.0f} | "
                f"lech cao {m.get('height_asymmetry', 0):.0f}px",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    if mild:
        cv2.putText(frame,
                    '! TAY LECH NHE — theo doi them (khong phai chuan doan)',
                    (10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 165, 255), 1)
    return frame


# ======================================================================
# THREAD 1 — CAMERA (~30fps, vẽ overlay trực tiếp)
# ======================================================================
def camera_worker():
    cap = cv2.VideoCapture(0)
    # NK-32: 720p — mặt người ngồi cách 3–4m tăng từ ~40–60px (640x480) lên
    # ~80–120px → MediaPipe khóa được; YOLO imgsz 480 giữ nguyên (letterbox,
    # chi phí KHÔNG đổi theo độ phân giải capture). BUFFERSIZE 1 = luôn xử
    # lý khung MỚI NHẤT, không dồn trễ khi đường vẽ chậm.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        STATE['cam_error'] = 'Không mở được webcam'
        return
    # NK-34: WARM-UP — C270 vừa mở cho khung ĐEN (độ sáng ~5/255,
    # auto-exposure chưa hội tụ ~3–6s, đo bằng probe: t0=5 → t+8s=137).
    # YOLO trên khung đen "bịa người" → xương chạy lung tung lúc mới mở.
    # Bỏ khung tối tối đa 6s; sáng sớm thì thoát ngay.
    warm_end = time.time() + 6.0
    while time.time() < warm_end:
        ok, fr = cap.read()
        if ok and fr is not None and \
                cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).mean() >= 20:
            break
        STATE['cam_error'] = 'Đang cải sáng camera (auto-exposure)...'
        time.sleep(0.1)
    STATE['cam_error'] = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            STATE['cam_error'] = 'Không đọc được webcam'
            STATE['camera_online'] = False
            time.sleep(0.2)
            continue
        STATE['cam_error'] = None
        STATE['camera_online'] = True
        frame = cv2.flip(frame, 1)
        # NK-32: toàn bộ khối vẽ bọc try/except — HUD/xương vẽ hỏng chỉ
        # mất overlay, KHÔNG được giết thread camera (đóng băng video).
        try:
            # vẽ khung xương YOLO từng khung (real-time)
            frame = draw_live_pose(frame)
            # NK-28: mặt REAL-TIME — detector riêng chạy mỗi khung chẵn:
            # mặt hiện ở đâu là khóa ngay (<100ms), không chờ chu kỳ 2s
            frame_idx += 1
            if frame_idx % 2 == 0:
                try:
                    STATE['face_live'] = FACE_LIVE.process_frame(frame)
                except Exception as e:
                    _rate_warn('face_live', f'FACE_LIVE: {e!r}', 30)
            # HUD mặt: ưu tiên live có mặt; vắng thì dùng kết quả chu kỳ 2s
            # (giữ HOLD 3s — mặt thoáng qua không nhấp nháy HUD)
            live = STATE.get('face_live')
            with ANALYSIS_LOCK:
                cached = STATE.get('hud_face')
            if live is not None and live.get('raw_landmarks') is not None:
                hud = live
            elif cached is not None and cached.get('raw_landmarks') is not None:
                hud = cached
            else:
                hud = live if live is not None else cached
            if hud:
                frame = draw_face_hud(frame, hud)
            # NK-34: overlay TRUNG THỰC khi ngồi sát — YOLO (COCO full-body)
            # không phát hiện được khi khung chỉ có đầu+vai; mặt vẫn khóa →
            # nói rõ hệ đang đo GÌ thay vì để khung trống như hỏng.
            if (STATE.get('pose_people', 0) == 0 and live is not None
                    and live.get('raw_landmarks') is not None):
                h_cam = frame.shape[0]
                cv2.putText(frame,
                            'NGOI SAT: chi thay MAT - he van do MAT + GIONG',
                            (12, h_cam - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(frame,
                            'Lui ra ~1m de YOLO thay than nguoi',
                            (12, h_cam - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 200, 255), 1, cv2.LINE_AA)
            draw_arm_panel(frame)
        except Exception as e:
            _rate_warn('cam_draw', f'overlay camera: {e!r}', 30)
        with CAM_LOCK:
            STATE['cam_frame'] = frame.copy()
        time.sleep(0.03)


# ======================================================================
# THREAD 2 — RADAR
# ======================================================================
def radar_worker():
    while True:
        with RADAR_LOCK:
            radar = STATE['radar']
        try:
            STATE['radar_result'] = radar.analyze(duration_s=0.6,
                                                  sample_hz=3.0)
        except Exception as e:
            STATE['radar_result'] = {'status': 'ERROR',
                                     'metrics': {'err': str(e)}}
        time.sleep(1.4)


# ======================================================================
# THREAD 3 — SPEECH (NK-32: module thứ 5 vào web — nghe liên tục)
# ======================================================================
SPEECH_LOCK = threading.Lock()
_SPEECH = None          # SpeechAnalysisModule — lazy init TRONG thread


def _speech_fresh():
    """Kết quả speech còn tươi (≤90s): trả (result, age_s); hết hạn/không có
    → None. Fusion nhận speech = tự cân lại trọng số (bỏ 0.20 speech)."""
    with SPEECH_LOCK:
        r = STATE.get('speech_result')
        ts = STATE.get('speech_ts') or 0.0
    if not r or not ts:
        return None
    age = time.time() - ts
    if age > 90:
        return None
    return r, round(age, 1)


def speech_worker():
    """Ghi 5s → phân tích 1 cửa sổ 5s → ngủ 5s (~12s/nhịp). Lazy init ở
    ĐẦU thread: nạp vosk 168MB + torch KHÔNG chặn mở dashboard; đúng 3 path
    production như app_family (KHÔNG dùng speech_torgo_full — NK-12)."""
    global _SPEECH
    while True:
        try:
            if _SPEECH is None:
                from detection.speech_module_v2 import SpeechAnalysisModule
                _SPEECH = SpeechAnalysisModule(
                    vosk_model_path=os.path.join(MODELS, 'vosk-model-vn-0.4'),
                    ml_model_path=os.path.join(
                        MODELS, 'speech_torgo_20260828_211130.pth'),
                    scaler_path=os.path.join(
                        MODELS, 'speech_torgo_20260828_211130_scaler.pkl'))
                _SPEECH.load_baseline(os.path.join(
                    PROJECT_ROOT, 'data', 'baselines', 'user_default.json'))
                STATE['speech_error'] = None
                STATE['speech_note'] = 'dang nghe…'
                print('[SPEECH] Sẵn sàng (vosk + ML) — nghe liên tục')
            audio = _SPEECH.record_audio(5)
            r = _SPEECH.predict_dysarthria(audio, 5)
            if r.get('status') == 'NO_SPEECH':
                # Nói khó là triệu chứng DAI DẲNG — KHÔNG xóa kết quả cũ,
                # giữ đến khi hết hạn 90s (fusion tự bỏ qua), chỉ ghi chú.
                with SPEECH_LOCK:
                    STATE['speech_note'] = 'im lang — dang nghe'
            else:
                with SPEECH_LOCK:
                    STATE['speech_result'] = r
                    STATE['speech_ts'] = time.time()
                    STATE['speech_note'] = ''
                STATE['speech_error'] = None
        except Exception as e:
            STATE['speech_error'] = str(e)
            with SPEECH_LOCK:
                STATE['speech_note'] = 'loi — thu lai sau 30s'
            _rate_warn('speech', f'speech_worker: {e!r}', 30)
            time.sleep(30)   # lỗi init/mic → chờ 30s (cắm mic sau tự lành)
            continue
        time.sleep(5)


# ======================================================================
# THREAD 4 — PHÂN TÍCH (mỗi 2 giây)
# ======================================================================
def analysis_worker():
    while True:
        with CAM_LOCK:
            frame = (None if STATE.get('cam_frame') is None
                     else STATE['cam_frame'].copy())
        if frame is not None and STATE['camera_online']:
            try:
                run_analysis_cycle(frame)
            except Exception as e:
                print(f'[ANALYSIS] {e}')
        time.sleep(2.0)


def run_analysis_cycle(frame):
    # ----- Cường độ chuyển động (L2 Context) -----
    gray = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
    motion = 0.0
    if STATE.get('prev_gray') is not None:
        motion = float(np.mean(np.abs(
            gray.astype(np.int16)
            - STATE['prev_gray'].astype(np.int16)))) / 50.0
        motion = min(motion, 1.0)
    STATE['prev_gray'] = gray
    STATE['motion'] = round(motion, 3)

    # ----- 3 module camera -----
    face_r = FACE.process_frame(frame)
    arm_r = ARM.detect_arm_weakness(frame)
    gait_r = GAIT.detect_gait_from_frame(frame)

    # ----- Radar từ thread riêng -----
    radar_r = STATE.get('radar_result') or {'status': 'NO_DATA',
                                            'metrics': {}}
    radar_r.setdefault('metrics', {})['simulation'] = \
        STATE['radar'].simulation
    # NK-32: cổng audio THẬT (ngữ nghĩa fall-AND-audio như app_family) —
    # radar chỉ nhận "âm thanh bất thường" khi speech còn tươi (≤90s)
    # VÀ điểm nói khó ≥ 30.
    sp_fresh = _speech_fresh()
    STATE['radar'].set_audio_flag(
        bool(sp_fresh and sp_fresh[0].get('speech_prob', 0) >= 30))

    ts = time.strftime('%H:%M:%S')
    STATE['radar_history'].append({
        'Thời điểm': ts, 'Trạng thái': radar_r['status'],
        'Dịch chuyển (cm)': radar_r['metrics'].get('position_change_cm', 0.0),
        'Bất động (s)': radar_r['metrics'].get('inactivity_s', 0.0),
        'Chế độ': 'SIM' if STATE['radar'].simulation else 'LIVE'})

    mods = {'face': face_r, 'arm': arm_r, 'gait': gait_r, 'radar': radar_r}
    if sp_fresh:
        mods['speech'] = sp_fresh[0]   # hết hạn → không đưa, fusion tự cân
    filtered, audit = DEFENSE.filter(mods)
    fused = FUSION.fuse(filtered)
    verdict = DEFENSE.verdict(fused['fused_score'])
    fused['trend'] = verdict['trend']

    # ----- NIHSS (NK-32 BUG A: tính TRƯỚC khối ALERT — trước đây biến nih
    # được dùng ở push alert TRƯỚC khi gán → UnboundLocalError MỌI chu kỳ
    # có cảnh báo, bị try/except ngoài nuốt → PROFILE.record + STATE +
    # heartbeat mobile bị BỎ đúng lúc báo động) -----
    nih = estimate_nihss(filtered)
    nih.update(calculate_nihss_ci(filtered, n_iter=300))

    # ----- ALERT -----
    alert_pushed = False
    if verdict['alert']:
        fused_alert = dict(fused)
        fused_alert['recommendation'] = \
            'GOI 115 — khong de nhan benh an/uong nuoc'
        res = ALERTER.process_fusion_result(fused_alert)
        if res.get('triggered'):
            STATE['last_alert'] = {
                'level': fused['risk_level'],
                'score': fused['fused_score'],
                'message': 'CAN BAO DOT QUY — kiem tra nguoi than!',
                'time': ts}
            HANDOFF.add_event('ALERT', f'Fusion {fused["fused_score"]}',
                              {'risk': fused['risk_level']})
            # NK-25: đẩy báo động sang app điện thoại (còi + rung + thẻ đỏ)
            push_mobile_event('alert', level=fused['risk_level'],
                              score=fused['fused_score'],
                              message='CAN BAO DOT QUY — kiem tra nguoi than!',
                              time=ts,
                              nihss_total=(nih.get('total')
                                           if isinstance(nih, dict) else None))
            alert_pushed = True

    # NK-32 BUG B: trước đây khối NIHSS + CYCLE bị LẶP 2 lần (CYCLE ghi
    # trùng, bootstrap NIHSS tính 2 lần/chu kỳ) — giờ ĐÚNG 1 event/chu kỳ.
    HANDOFF.add_event('CYCLE', f'score={fused["fused_score"]} '
                               f'risk={fused["risk_level"]}')

    # ----- Chế độ học 3 ngày (đặc điểm riêng: gù, cong lung, mép quen...) -----
    p_st = PROFILE.record(face_r, gait_score=gait_r.get('gait_prob', 0.0),
                          motion=motion)
    if p_st['done'] and not p_st['applied']:
        base = PROFILE.baseline()
        if base:
            DEFENSE.calibrate(face_asym=base.get('mouth_ratio'),
                              arm_asym=None)
            PROFILE.mark_applied()
            print(f"[PROFILE] Đủ {PROFILE.DAYS} ngày — baseline cá nhân "
                  f"áp dụng: {base}")

    # ----- Speech entry cho UI (đọc 1 lần dưới SPEECH_LOCK) -----
    with SPEECH_LOCK:
        sr = STATE.get('speech_result')
        s_ts = STATE.get('speech_ts') or 0.0
        s_note = STATE.get('speech_note')
        s_err = STATE.get('speech_error')
    if sr:
        s_age = round(time.time() - s_ts, 1)
        s_m = sr.get('metrics') or {}
        speech_mod = {
            'status': sr.get('status', '?'),
            'score': round(float(sr.get('speech_prob', 0) or 0), 1),
            'wpm': s_m.get('wpm'), 'words': s_m.get('word_count'),
            'transcript': s_m.get('transcript'),
            'age_s': s_age, 'stale': s_age > 90,
            'note': s_note, 'error': s_err}
    else:
        speech_mod = {'status': 'NO_DATA', 'score': 0.0, 'wpm': None,
                      'words': None, 'transcript': None, 'age_s': None,
                      'stale': True, 'note': s_note, 'error': s_err}

    with ANALYSIS_LOCK:
        STATE['hud_face'] = dict(face_r)
        STATE['arm_metrics'] = arm_r.get('metrics') or {}
        STATE['modules'] = {
            'face': {'status': face_r.get('status', '?'),
                     'score': round(float(face_r.get('score', 0) or 0), 1)},
            'arm': {'status': arm_r.get('status', '?'),
                    'score': round(float(arm_r.get('arm_prob', 0) or 0), 1)},
            'gait': {'status': gait_r.get('status', '?'),
                     'score': round(float(gait_r.get('gait_prob', 0) or 0),
                                    1)},
            'radar': {'status': radar_r['status'],
                      'score': round(float(
                          radar_r['metrics'].get('fall_prob', 0) or 0), 1)},
            'speech': speech_mod,
        }
        STATE['fused_score'] = round(float(fused['fused_score']), 1)
        STATE['risk_level'] = fused['risk_level']
        STATE['verdict'] = verdict
        STATE['nihss'] = nih
        STATE['analysis_count'] += 1
    # NK-25: nhịp tim cho app điện thoại (heartbeat 2s — app mất nhịp
    # >15s = mất kết nối với máy chủ → tự báo trên điện thoại)
    push_mobile_event('status', risk_level=fused['risk_level'],
                      fused_score=round(float(fused['fused_score']), 1),
                      camera_online=STATE['camera_online'],
                      analysis_count=STATE['analysis_count'],
                      radar_mode=STATE['radar_mode'])


# ======================================================================
# FASTAPI APP
# ======================================================================
app = FastAPI(title='Golden-Watch PSCS', docs_url=None, redoc_url=None)


def mjpeg_boundary():
    while True:
        with CAM_LOCK:
            frame = (None if STATE.get('cam_frame') is None
                     else STATE['cam_frame'].copy())
        if frame is None:
            # đang khởi động camera — khung chờ
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, 'DANG KET NOI CAMERA...',
                        (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
        ok, jpg = cv2.imencode('.jpg', frame,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + jpg.tobytes() + b'\r\n')
        time.sleep(0.04)   # ~25 fps mục tiêu (nguồn 30fps)


@app.get('/video.mjpg')
def video_mjpg():
    return StreamingResponse(
        mjpeg_boundary(),
        media_type='multipart/x-mixed-replace; boundary=frame')


@app.get('/api/state')
def api_state():
    with ANALYSIS_LOCK:
        snap = {k: STATE[k] for k in ('modules', 'fused_score', 'risk_level',
                                      'verdict', 'nihss', 'last_alert',
                                      'motion', 'camera_online',
                                      'analysis_count', 'patient',
                                      'radar_mode')}
    snap['radar_history'] = list(STATE['radar_history'])[-10:]
    snap['alerts'] = ALERTER.get_recent_alerts(8)
    snap['defense_stats'] = DEFENSE.get_stats()
    snap['profile'] = PROFILE.status()
    snap['time'] = time.strftime('%H:%M:%S')
    return JSONResponse(snap)


@app.post('/api/radar')
async def api_radar(req: Request):
    """Đổi radar: {"simulation": true, "scenario": "normal|fall|wander",
    "port": "COM3"(khi simulation=false)}"""
    body = await req.json()
    sim = bool(body.get('simulation', True))
    device = body.get('port')
    ok_port = bool(device and str(device).upper().startswith('COM'))
    if getattr(STATE['radar'], 'serial', None) is not None:
        try:
            STATE['radar'].serial.close()
        except Exception:
            pass
    STATE['radar'] = RadarModule(
        simulation=sim,
        port=None if (sim or not ok_port) else str(device),
        sim_scenario=body.get('scenario', 'normal'))
    STATE['radar_mode'] = ('🟡 SIMULATION' if sim else
                           f'🟢 LIVE @ {device}' if ok_port
                           else '🟡 SIMULATION (port lỗi)')
    STATE['radar_history'].clear()
    return {'ok': True, 'mode': STATE['radar_mode']}


@app.post('/patient')
async def patient_post(req: Request):
    form = await req.form()
    for k in ('name', 'age', 'gender', 'phone', 'address'):
        if k in form:
            STATE['patient'][k] = str(form[k])
    return RedirectResponse('/', status_code=303)


@app.post('/alert/test')
async def alert_test():
    ALERTER.send_alert(95, 'Test he thong bao dong (web)',
                       level='EMERGENCY', source='web-test')
    push_mobile_event('alert', level='EMERGENCY', score=95,
                      message='Test he thong bao dong (web)',
                      time=time.strftime('%H:%M:%S'))
    return {'ok': True}


# ======================================================================
# MOBILE ALERT (NK-25) — app điện thoại PWA qua LAN/hotspot, KHÔNG internet
# ======================================================================
@app.websocket('/ws')
async def ws_events(ws: WebSocket):
    """Kênh sự kiện thời gian thực: snapshot đầu tiên → các sự kiện mới
    (status mỗi 2s · alert khi báo động). Client ngắt → thoát sạch."""
    await ws.accept()
    last = 0
    try:
        await ws.send_json(build_mobile_snapshot())
        while True:
            with MOBILE_LOCK:
                pending = [e for e in STATE['mobile_events']
                           if e['seq'] > last]
            for ev in pending:
                await ws.send_json(ev)
                last = ev['seq']
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass


@app.get('/mobile', response_class=HTMLResponse)
def mobile_page():
    return MOBILE_HTML


@app.get('/mobile_manifest.json')
def mobile_manifest():
    return JSONResponse({
        'name': 'Golden-Watch Cảnh báo',
        'short_name': 'Golden-Watch',
        'start_url': '/mobile',
        'display': 'standalone',
        'background_color': '#0d1117',
        'theme_color': '#c81e1e',
        'icons': [{'src': '/mobile/icon-192.png', 'sizes': '192x192',
                   'type': 'image/png'},
                  {'src': '/mobile/icon-512.png', 'sizes': '512x512',
                   'type': 'image/png'}]})


def _mobile_icon(size):
    """Icon đồng hồ trắng trên nền đỏ đậm — sinh 1 lần, cache ra exports/."""
    path = os.path.join(PROJECT_ROOT, 'exports', f'mobile_icon_{size}.png')
    if not os.path.exists(path):
        img = np.full((size, size, 3), (30, 30, 200), dtype=np.uint8)
        img[:, :size // 8] = (20, 20, 160)
        c, r = size // 2, int(size * 0.36)
        cv2.circle(img, (c, c), r, (255, 255, 255), max(2, size // 24),
                   cv2.LINE_AA)
        cv2.circle(img, (c, c), max(3, size // 40), (255, 255, 255), -1,
                   cv2.LINE_AA)
        cv2.line(img, (c, c), (c, c - int(r * 0.62)), (255, 255, 255),
                 max(2, size // 28), cv2.LINE_AA)
        cv2.line(img, (c, c), (c + int(r * 0.45), c + int(r * 0.25)),
                 (255, 255, 255), max(2, size // 28), cv2.LINE_AA)
        cv2.imwrite(path, img)
    return path


@app.get('/mobile/icon-192.png')
def mobile_icon_192():
    return FileResponse(_mobile_icon(192), media_type='image/png')


@app.get('/mobile/icon-512.png')
def mobile_icon_512():
    return FileResponse(_mobile_icon(512), media_type='image/png')


@app.get('/mobile/qr')
def mobile_qr():
    """QR mở app trên điện thoại (dùng IP LAN thật — hotspot hoặc wifi nhà)."""
    import qrcode
    import io
    url = f'http://{detect_lan_ip()}:5001/mobile'
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return Response(buf.getvalue(), media_type='image/png')


@app.post('/mobile/ack')
async def mobile_ack():
    """Điện thoại bấm 'TÔI ỔN' — ghi vào log bàn giao để đối chiếu."""
    HANDOFF.add_event('MOBILE_ACK', 'Người thân xác nhận trên app điện thoại',
                      {'time': time.strftime('%H:%M:%S')})
    print('[MOBILE] Người thân đã xác nhận "TÔI ỔN" trên app')
    return {'ok': True}


@app.get('/handoff/qr')
def handoff_qr():
    nih = STATE.get('nihss') or {'total': 0}
    fused = {'fused_score': STATE['fused_score'],
             'risk_level': STATE['risk_level']}
    report = HANDOFF.build_report(fused, nih)
    png, _data = HANDOFF.generate_qr(report)   # trả về (path, data)
    return FileResponse(png, media_type='image/png')


@app.get('/handoff/pdf')
def handoff_pdf():
    nih = STATE.get('nihss') or {'total': 0}
    fused = {'fused_score': STATE['fused_score'],
             'risk_level': STATE['risk_level']}
    report = HANDOFF.build_report(fused, nih)
    pdf = HANDOFF.generate_pdf(report)
    return FileResponse(pdf, media_type='application/pdf')


@app.get('/', response_class=HTMLResponse)
def index():
    return PAGE_HTML


# ======================================================================
# TRANG WEB (inline CSS/JS — chạy offline, không cần mạng)
# ======================================================================
PAGE_HTML = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Golden-Watch — Giám sát thời gian vàng</title>
<style>
  :root{--bg:#0d1117;--card:#161b22;--line:#30363d;--tx:#e6edf3;
        --mut:#8b949e;--grn:#2ea043;--ylw:#d29922;--red:#c81e1e}
  *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);
    font-family:'Segoe UI',Arial,sans-serif}
  header{display:flex;align-items:center;gap:16px;padding:10px 20px;
    background:var(--card);border-bottom:1px solid var(--line)}
  header h1{font-size:20px;margin:0}
  #score{font-size:26px;font-weight:700}
  .badge{padding:6px 14px;border-radius:10px;font-weight:700;color:#fff;
    background:#555}
  main{display:grid;grid-template-columns:1.6fr 1fr;gap:14px;
    padding:14px 20px}
  .card{background:var(--card);border:1px solid var(--line);
    border-radius:10px;padding:12px}
  img#cam{width:100%;border-radius:8px;background:#000}
  .cap{color:var(--mut);font-size:12px;margin-top:6px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  td,th{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left}
  .mod{display:flex;justify-content:space-between;padding:6px 10px;
    border:1px solid var(--line);border-radius:8px;margin-bottom:6px}
  .st-NORMAL{color:var(--grn);font-weight:700}
  .st-WARNING{color:var(--ylw);font-weight:700}
  .st-DANGER,.st-EMERGENCY{color:var(--red);font-weight:700}
  #banner{display:none;background:var(--red);color:#fff;padding:14px 20px;
    font-size:18px;font-weight:800;text-align:center}
  .row{display:flex;gap:8px;flex-wrap:wrap}
  input,select{background:#0d1117;color:var(--tx);border:1px solid var(--line);
    border-radius:6px;padding:6px 8px;width:100%}
  label{font-size:12px;color:var(--mut)}
  button{background:#21262d;color:var(--tx);border:1px solid var(--line);
    border-radius:6px;padding:8px 14px;cursor:pointer;font-weight:600}
  button.primary{background:#1f6feb;border-color:#1f6feb;color:#fff}
  button.danger{background:var(--red);border-color:var(--red);color:#fff}
  a{color:#58a6ff}
  footer{color:var(--mut);text-align:center;padding:10px;font-size:12px}
</style></head><body>
<div id="banner">🚨 CẢNH BÁO ĐỘT QUỴ — KIỂM TRA NGƯỜI THÂN NGAY! GỌI 115</div>
<div id="learn" style="background:#1f4d2e;color:#d6f5df;padding:8px 20px;
  font-size:13px;display:none">🧠 Chế độ học đặc điểm cá nhân — <span id="learntxt"></span></div>
<header>
  <h1>⏱️ Golden-Watch</h1>
  <span>Điểm nguy cơ <span id="score">—</span>/100</span>
  <span class="badge" id="risk">—</span>
  <span style="margin-left:auto;color:var(--mut)" id="clock"></span>
</header>
<main>
  <div>
    <div class="card">
      <img id="cam" src="/video.mjpg" alt="camera">
      <div class="cap">🔴 TRỰC TIẾP ~25fps · khung xương YOLO từng khung ·
        méo mặt MediaPipe + HUD cập nhật 2s · radar <span id="radarmode">—</span></div>
    </div>
    <div class="card" style="margin-top:14px">
      <b>📡 Radar LD2450 — 10 dòng gần nhất</b>
      <table id="radartab"><tr><th>Thời điểm</th><th>Trạng thái</th>
        <th>Dịch chuyển (cm)</th><th>Bất động (s)</th><th>Chế độ</th></tr></table>
      <div class="row" style="margin-top:8px">
        <select id="scenario">
          <option value="normal">SIM: bình thường</option>
          <option value="fall">SIM: ngã</option>
          <option value="wander">SIM: đi lang thang</option>
        </select>
        <button onclick="applyRadar(true)">Áp dụng SIM</button>
        <input id="port" placeholder="COMx (radar thật @256000)"
               style="width:180px">
        <button onclick="applyRadar(false)">Áp dụng LIVE</button>
      </div>
    </div>
  </div>
  <div>
    <div class="card">
      <b>📷 Trạng thái module</b>
      <div id="mods" style="margin-top:8px"></div>
      <div class="cap">Verdict: <span id="verdict">—</span> ·
        Trend: <span id="trend">—</span> ·
        Chu kỳ phân tích: <span id="cycles">0</span></div>
    </div>
    <div class="card" style="margin-top:14px">
      <b>🗣️ Giọng nói — liên tục</b>
      <div id="speech" style="margin-top:6px;font-size:13px;line-height:1.6">
        <span style="color:var(--mut)">Đang khởi động (nạp vosk + ML vài
        giây)…</span></div>
      <div class="cap">Ghi 5s mỗi ~12s · kết quả giữ 90s rồi fusion tự bỏ ·
        Nói khó ≥30% = cảnh giác (KHÔNG chẩn đoán)</div>
    </div>
    <div class="card" style="margin-top:14px">
      <b>🧩 Ước tính NIHSS (4 item — tối đa 13)</b>
      <div style="font-size:24px;margin-top:6px" id="nihss">—</div>
      <div class="cap" id="nihssnote">UỚC LƯỢNG HỖ TRỢ — bác sĩ chấm chuẩn</div>
    </div>
    <div class="card" style="margin-top:14px">
      <b>🚨 Tin nhắn báo động</b>
      <div id="alerts" style="margin-top:6px;font-size:13px"></div>
      <div class="row" style="margin-top:8px">
        <button class="danger" onclick="fetch('/alert/test',{method:'POST'})
          .then(()=>alert('Đã phát cảnh báo thử'))">🔊 Test còi + cảnh báo</button>
        <a href="/handoff/qr" target="_blank"><button>QR bàn giao</button></a>
        <a href="/handoff/pdf" target="_blank"><button>PDF bệnh viện</button></a>
      </div>
    </div>
    <div class="card" style="margin-top:14px">
      <b>📱 App cảnh báo trên điện thoại</b>
      <div class="row" style="margin-top:8px;align-items:center">
        <img src="/mobile/qr" alt="QR app điện thoại"
             style="width:130px;height:130px;background:#fff;
                    border-radius:8px;padding:4px">
        <div style="flex:1;min-width:200px">
          <div style="font-size:13px;line-height:1.6">
            1. Chạy server với cờ <code>--lan</code> (hotspot laptop hoặc
            wifi nhà — KHÔNG cần internet)<br>
            2. Điện thoại quét QR này (hoặc mở
            <code>http://&lt;IP-laptop&gt;:5001/mobile</code>)<br>
            3. Chrome → ⋮ → <b>Thêm vào Màn hình chính</b> = cài app
          </div>
        </div>
      </div>
    </div>
    <div class="card" style="margin-top:14px">
      <b>👤 Thông tin người bệnh</b>
      <form action="/patient" method="post" style="margin-top:8px">
        <div class="row"><div style="flex:2"><label>Họ tên</label>
          <input name="name" id="p_name"></div>
        <div style="flex:1"><label>Tuổi</label><input name="age" id="p_age"></div>
        <div style="flex:1"><label>Giới</label>
          <select name="gender" id="p_gender">
            <option>Nam</option><option>Nữ</option></select></div></div>
        <div class="row" style="margin-top:6px">
          <div style="flex:1"><label>Điện thoại</label>
            <input name="phone" id="p_phone"></div>
          <div style="flex:2"><label>Địa chỉ</label>
            <input name="address" id="p_address"></div></div>
        <button class="primary" style="margin-top:8px">Lưu</button>
      </form>
    </div>
  </div>
</main>
<footer>KHÔNG PHẢN CHẨN ĐOÁN — công cụ hỗ trợ phát hiện sớm.
  Khẩn cấp: GỌI 115 · Golden-Watch PSCS (nghiên cứu, chạy local)</footer>
<script>
function esc(s){return (s??'').toString().replace(/[<>&]/g,
  c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}
async function tick(){
  try{
    const s = await (await fetch('/api/state')).json();
    document.getElementById('clock').textContent = s.time;
    document.getElementById('score').textContent = s.fused_score;
    const r = document.getElementById('risk');
    r.textContent = s.risk_level;
    r.style.background = {NORMAL:'#1a7a3a',MONITOR:'#b8860b',
      WARNING:'#d97706',EMERGENCY:'#c81e1e'}[s.risk_level]||'#555';
    document.getElementById('banner').style.display =
      (s.risk_level==='EMERGENCY'||s.risk_level==='WARNING')?'block':'none';
    const p = s.profile||{};
    const learn = document.getElementById('learn');
    if (p.done && p.applied){
      learn.style.display='block';
      document.getElementById('learntxt').textContent =
        'ĐÃ HỌC XONG — hệ thống đã nắm đặc điểm riêng của người dùng '
        + '(gù, cong cột sống, mép thói quen...) và cá nhân hóa ngưỡng.';
    } else if (!p.applied){
      learn.style.display='block';
      document.getElementById('learntxt').textContent =
        `Ngày ${p.day||1}/3 — đã ghi ${p.samples_today||0} mẫu hôm nay. `
        + 'Hệ thống đang quan sát để nhận diện đặc điểm riêng (gù lưng, '
        + 'cong cột sống, nhech mép thói quen...), chưa cá nhân hóa.';
    } else { learn.style.display='none'; }
    document.getElementById('radarmode').textContent = s.radar_mode;
    document.getElementById('verdict').textContent =
      (s.verdict.alert?'BẮT':'chặn') + ' — ' + s.verdict.reason;
    document.getElementById('trend').textContent = s.verdict.trend;
    document.getElementById('cycles').textContent = s.analysis_count;
    const mods = document.getElementById('mods'); mods.innerHTML = '';
    const name = {face:'Méo mặt (MediaPipe)',arm:'Tay yếu (YOLO)',
      gait:'Dáng đi (YOLO)',radar:'Radar LD2450',speech:'Giọng nói'};
    for (const k in s.modules){
      const m = s.modules[k];
      mods.innerHTML += `<div class="mod"><span>${name[k]||k}</span>
        <span><span class="st-${esc(m.status)}">${esc(m.status)}</span>
        &nbsp;${m.score}/100</span></div>`;
    }
    const sp = (s.modules||{}).speech;
    if (sp){
      const stale = sp.stale;
      const col = stale ? 'var(--mut)' :
        ({NORMAL:'var(--grn)',WARNING:'var(--ylw)',
          DANGER:'var(--red)'}[sp.status]||'var(--mut)');
      let h = '<span style="color:'+col+';font-weight:700">'+esc(sp.status)
        +(stale?' (hết hạn — fusion bỏ)':'')+'</span> — điểm <b>'
        +(sp.score??0)+'/100</b> · '+(sp.wpm??'—')+' từ/phút · '
        +(sp.words??'—')+' từ';
      if (sp.age_s!=null) h += ' · cách đây '+sp.age_s+'s';
      if (sp.note) h += '<br><span style="color:var(--mut)">'
        +esc(sp.note)+'</span>';
      if (sp.error) h += '<br><span style="color:var(--red)">Lỗi: '
        +esc(sp.error)+'</span>';
      if (sp.transcript) h += '<br>🎙️ “'+esc(sp.transcript)+'”';
      document.getElementById('speech').innerHTML = h;
    }
    if (s.nihss){
      document.getElementById('nihss').textContent =
        `${s.nihss.total} ± ${s.nihss.margin} (CI ${s.nihss.ci_low}–` +
        `${s.nihss.ci_high}) / 13`;
      document.getElementById('nihssnote').textContent =
        'Thiếu item: ' + (s.nihss.items_missing?.join(', ')||'không');
    }
    const al = document.getElementById('alerts');
    al.innerHTML = (s.alerts||[]).map(a =>
      `<div>⚠️ [${esc(a.level)}] ${esc(a.time||'')} — ${esc(a.message)}</div>`
    ).join('') || '<span style="color:var(--mut)">Chưa có cảnh báo</span>';
    const rt = document.getElementById('radartab');
    rt.innerHTML = rt.rows[0].outerHTML + (s.radar_history||[]).map(r =>
      `<tr><td>${esc(r['Thời điểm'])}</td><td>${esc(r['Trạng thái'])}</td>
       <td>${esc(r['Dịch chuyển (cm)'])}</td><td>${esc(r['Bất động (s)'])}</td>
       <td>${esc(r['Chế độ'])}</td></tr>`).join('');
    if (s.patient.name && !document.getElementById('p_name').value){
      for (const k of ['name','age','gender','phone','address'])
        document.getElementById('p_'+k).value = s.patient[k]||'';
    }
  }catch(e){ console.error(e); }
}
async function applyRadar(sim){
  const body = {simulation: sim,
                scenario: document.getElementById('scenario').value,
                port: document.getElementById('port').value};
  const r = await (await fetch('/api/radar',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)})).json();
  alert(r.mode);
}
setInterval(tick, 1000); tick();
</script></body></html>"""


# ======================================================================
# TRANG APP ĐIỆN THOẠI (PWA — mở /mobile → 'Thêm vào Màn hình chính')
# ======================================================================
MOBILE_HTML = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,
  user-scalable=no">
<link rel="manifest" href="/mobile_manifest.json">
<meta name="theme-color" content="#c81e1e">
<title>Golden-Watch — Cảnh báo</title>
<style>
  :root{--bg:#0d1117;--card:#161b22;--line:#30363d;--tx:#e6edf3;
    --mut:#8b949e;--grn:#2ea043;--ylw:#d29922;--red:#e5484d}
  *{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
  body{margin:0;background:var(--bg);color:var(--tx);
    font-family:'Segoe UI',Arial,sans-serif}
  #conn{position:fixed;top:0;left:0;right:0;padding:8px 14px;font-size:13px;
    font-weight:700;text-align:center;z-index:5}
  .on{background:#123d1f;color:#7ee2a0}.off{background:#5c1015;color:#ffb4b6}
  main{padding:56px 14px 20px;max-width:520px;margin:0 auto}
  .pill{display:inline-block;padding:6px 16px;border-radius:20px;
    font-weight:800;color:#fff;background:#444}
  .big{font-size:64px;font-weight:800;line-height:1;margin:10px 0}
  .card{background:var(--card);border:1px solid var(--line);
    border-radius:14px;padding:14px;margin-top:12px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}
  .mod{border:1px solid var(--line);border-radius:10px;padding:8px;
    font-size:13px;display:flex;justify-content:space-between}
  .g{color:var(--grn);font-weight:800}.y{color:var(--ylw);font-weight:800}
  .r{color:var(--red);font-weight:800}
  button{border:0;border-radius:12px;padding:16px;font-size:17px;
    font-weight:800;width:100%;margin-top:10px;color:#fff;cursor:pointer}
  .b-ack{background:#1a7a3a}.b-115{background:#c81e1e}
  .b-cam{background:#1f6feb}.b-perm{background:#444;padding:10px;
    font-size:14px}
  #overlay{position:fixed;inset:0;background:rgba(160,10,10,.97);z-index:9;
    display:none;overflow:auto;padding:24px 18px;text-align:center}
  #ovl-ylw{position:fixed;inset:0;background:rgba(120,80,0,.97);z-index:8;
    display:none;overflow:auto;padding:24px 18px;text-align:center}
  #overlay h1,#ovl-ylw h1{font-size:26px;margin:6px 0}
  #overlay .s,#ovl-ylw .s{font-size:16px;color:#ffe;min-height:22px}
  .tip{color:var(--mut);font-size:12px;margin-top:14px;line-height:1.6}
  #hist{font-size:13px;line-height:1.7;min-height:20px}
</style></head><body>
<div id="conn" class="off">⏳ ĐANG KẾT NỐI VỚI MÁY CHỦ…</div>
<main>
  <div style="text-align:center">
    <span class="pill" id="risk">—</span>
    <div class="big"><span id="score">—</span><span style="font-size:22px;
      color:var(--mut)">/100</span></div>
    <div style="color:var(--mut)" id="pt">Chưa nhập người bệnh</div>
  </div>
  <div class="card"><b>📷 5 module — trạng thái</b>
    <div class="grid" id="mods"></div>
    <div style="margin-top:8px;color:var(--mut);font-size:13px">
      NIHSS ước tính: <b id="nihss">—</b> · Chu kỳ: <b id="cyc">0</b> ·
      Radar: <span id="radarmode">—</span></div>
  </div>
  <div class="card"><b>🔔 Lịch cảnh báo</b><div id="hist" style="margin-top:6px">
    <span style="color:var(--mut)">Chưa có</span></div>
    <button class="b-perm" onclick="askPerm()">🔓 Bật thông báo màn hình khóa
      (cho phép 1 lần)</button>
  </div>
  <div class="tip">📲 Cài như app: Chrome → nút ⋮ → <b>Thêm vào Màn hình
    chính</b>. Hoạt động qua hotspot laptop — KHÔNG cần internet.<br>
    ⚠️ KHÔNG PHẢN CHẨN ĐOÁN — khẩn cấp GỌI 115.</div>
</main>
<div id="ovl-ylw">
  <h1>⚠️ NGUY CƠ TRUNG BÌNH</h1><div class="s" id="ylw-s"></div>
  <button class="b-ack" onclick="ack()">✓ TÔI ĐÃ BIẾT — TẮT CẢNH BÁO</button>
</div>
<div id="overlay">
  <h1>🚨 CẢNH BÁO ĐỘT QUỴ</h1>
  <div class="s" id="ovl-s"></div>
  <div style="font-size:20px;margin-top:8px" id="ovl-nihss"></div>
  <button class="b-115" onclick="location.href='tel:115'">📞 GỌI 115 NGAY</button>
  <button class="b-ack" onclick="ack()">✓ TÔI ỔN — TẮT CÒI</button>
  <button class="b-cam" onclick="window.open('/video.mjpg')">📷 XEM CAMERA</button>
</div>
<script>
let ws=null,seq=0,lastEv=Date.now(),siren=null,vibT=null,alarmLv=null;
const $=id=>document.getElementById(id);
function esc(s){return (s??'').toString().replace(/[<>&]/g,
  c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}
function beep(f,d,t0){const o=ac.createOscillator(),g=ac.createGain();
  o.frequency.value=f;o.type='square';g.gain.value=.12;
  o.connect(g);g.connect(ac.destination);
  o.start(t0);o.stop(t0+d/1000)}
let ac=null;
function sirenStart(){if(siren)return;ac=ac||new (window.AudioContext||
  window.webkitAudioContext)();const loop=()=>{const t=ac.currentTime;
  beep(660,300,t);beep(880,300,t+.3);beep(660,300,t+.6);beep(880,300,t+.9)};
  loop();siren=setInterval(loop,1300);
  vibT=setInterval(()=>navigator.vibrate&&navigator.vibrate([400,200,400,200,
    400,800]),2400)}
function silence(){if(siren){clearInterval(siren);siren=null}
  if(vibT){clearInterval(vibT);vibT=null}
  navigator.vibrate&&navigator.vibrate(0)}
function notify(title,body){if(Notification&&Notification.permission===
  'granted'){try{navigator.serviceWorker&&0;
    new Notification(title,{body:body,tag:'gw-alert'})}catch(e){}}}
function askPerm(){if(window.Notification){Notification.requestPermission()
  .then(p=>alert(p==='granted'?'Đã bật — báo động sẽ hiện cả khi tắt màn hình'
  :'Chưa được phép — vẫn có còi + rung khi app mở'))}}
function showAlert(ev){
  if(ev.level==='EMERGENCY'){
    $('ovl-s').textContent=(ev.message||'')+' — điểm '+ev.score+'/100 · '
      +ev.time;
    $('ovl-nihss').textContent='NIHSS ước tính: '
      +(ev.nihss_total??'…')+' / 13';
    $('overlay').style.display='block';$('ovl-ylw').style.display='none';
    sirenStart();
    notify('🚨 CẢNH BÁO ĐỘT QUỴ','Điểm '+ev.score+'/100 — kiểm tra người thân!')
  } else {
    $('ylw-s').textContent='Điểm '+ev.score+'/100 lúc '+ev.time
      +' — nên đến gặp người thân kiểm tra.';
    $('ovl-ylw').style.display='block';$('overlay').style.display='none';
    ac=ac||new (window.AudioContext||window.webkitAudioContext)();
    beep(520,500,ac.currentTime);
    notify('⚠️ Golden-Watch: nguy cơ trung bình','Điểm '+ev.score+'/100');
    setTimeout(()=>{$('ovl-ylw').style.display='none'},30000)
  }
  addHist(ev)}
function addHist(ev){const h=$('hist');
  if(h.textContent.indexOf('Chưa có')>=0)h.textContent='';
  h.innerHTML='⚠️ ['+esc(ev.level)+'] '+esc(ev.time)+' — điểm '
    +esc(ev.score)+'<br>'+h.innerHTML}
function ack(){silence();$('overlay').style.display='none';
  $('ovl-ylw').style.display='none';
  fetch('/mobile/ack',{method:'POST'})}
function paint(st){ // vẽ trạng thái thường
  $('score').textContent=st.fused_score??'—';
  const r=$('risk');r.textContent=st.risk_level||'—';
  r.style.background={NORMAL:'#1a7a3a',MONITOR:'#b8860b',WARNING:'#d97706',
    EMERGENCY:'#c81e1e'}[st.risk_level]||'#555';
  const p=st.patient||{};
  $('pt').textContent=p.name?('Người giám sát: '+p.name
    +(p.age?(' · '+p.age+' tuổi'):'')):'Chưa nhập người bệnh';
  const nm={face:'Méo mặt',arm:'Tay yếu',gait:'Dáng đi',radar:'Radar',
    speech:'Giong noi'};
  const cl=s=>s==='NORMAL'?'g':(s==='WARNING'||s==='MONITOR')?'y':'r';
  let h='';for(const k in st.modules||{}){const m=st.modules[k];
    h+='<div class="mod"><span>'+esc(nm[k]||k)+'</span><span class="'
    +cl(m.status)+'">'+esc(m.status)+' '+m.score+'</span></div>'}
  $('mods').innerHTML=h||'<i style="color:var(--mut)">Chờ chu kỳ phân tích…</i>';
  if(st.nihss)$('nihss').textContent=(st.nihss.total??'—')+' ± '
    +(st.nihss.margin??'?')+' / 13';
  $('cyc').textContent=st.analysis_count??0;
  $('radarmode').textContent=st.radar_mode||'—'}
function connect(){
  const proto=location.protocol==='https:'?'wss':'ws';
  ws=new WebSocket(proto+'://'+location.host+'/ws');
  ws.onopen=()=>{$('conn').className='on';
    $('conn').textContent='🟢 ĐANG KẾT NỐI VỚI MÁY CHỦ — '+location.host};
  ws.onmessage=m=>{lastEv=Date.now();let ev;
    try{ev=JSON.parse(m.data)}catch(e){return}
    if(ev.type==='snapshot'){paint(ev);
      if(ev.last_alert)addHist(ev.last_alert);return}
    if(ev.type==='status'){paint(ev);return}
    if(ev.type==='alert'||ev.type==='alert_nihss'){
      if(ev.type==='alert_nihss'&&$('overlay').style.display==='block'){
        $('ovl-nihss').textContent='NIHSS ước tính: '
          +(ev.nihss_total??'…')+' ± '+(ev.nihss_margin??'?')+' / 13';return}
      showAlert(ev)}};
  ws.onclose=()=>{$('conn').className='off';
    $('conn').textContent='🔴 MẤT KẾT NỐI VỚI MÁY CHỦ — kiểm tra laptop!';
    silence();setTimeout(connect,2000)};
  ws.onerror=()=>ws.close()}
// watchdog: quá 15s không có nhịp tim nào → coi như mất kết nối
setInterval(()=>{if(Date.now()-lastEv>15000&&$('conn').className==='on'){
  $('conn').className='off';
  $('conn').textContent='🔴 MẤT NHỊP TÍN HIỆU (>15s) — kiểm tra laptop!';
  silence()}},3000);
connect();
</script></body></html>"""


# ======================================================================
# MAIN
# ======================================================================
def _port_busy(port=5001):
    """Kiểm tra cổng đã có process nào nghe chưa (tránh lỗi bind ầm ĩ)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1', port)) == 0


def main():
    if _port_busy(5001):
        print('=' * 64)
        print('  LỖI: Cổng 5001 ĐANG ĐƯỢC DÙNG — có thể server cũ chưa tắt.')
        print('  Cách tắt bản cũ (chọn 1):')
        print('    1) Tìm cửa sổ server cũ nhấn Ctrl+C')
        print('    2) Hoặc chạy lệnh:')
        print('       netstat -ano | findstr :5001')
        print('       taskkill /F /PID <số PID ở cột cuối>')
        print('=' * 64)
        sys.exit(1)

    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=radar_worker, daemon=True).start()
    threading.Thread(target=speech_worker, daemon=True).start()
    threading.Thread(target=analysis_worker, daemon=True).start()

    import uvicorn
    lan_ip = detect_lan_ip()
    print('=' * 64)
    print('  GOLDEN-WATCH WEB — mở trình duyệt:  http://localhost:5001')
    print('  Video MJPEG (mở VLC được): http://localhost:5001/video.mjpg')
    if HOST == '0.0.0.0':
        print('  CHẾ ĐỘ --lan: app điện thoại mở  '
              f'http://{lan_ip}:5001/mobile')
        print('  (Dashboard cũng có QR ở thẻ "App điện thoại")')
        print('  ⚠️ Server nghe trên MẠNG CỤC BỘ — chỉ dùng hotspot/wifi nhà')
    else:
        print('  Chỉ nghe trên 127.0.0.1 — KHÔNG mở ra mạng ngoài.')
        print('  Bật app điện thoại: chạy lại với cờ  --lan')
    print('  Ctrl+C để dừng · KHÔNG PHẢN CHẨN ĐOÁN — khẩn cấp GỌI 115')
    print('=' * 64)
    uvicorn.run(app, host=HOST, port=5001, log_level='warning')


if __name__ == '__main__':
    main()
