# -*- coding: utf-8 -*-
"""
GOLDEN-WATCH — WEB SERVER GIÁM SÁT THỜI GIAN THỰC (FastAPI, chạy LOCAL)

Lý do thay Streamlit cho video: Streamlit rerun cả script mỗi tick → video
giật; FastAPI + MJPEG stream = đúng kiểu web camera giám sát (20–30 fps).

Kiến trúc 3 luồng trong 1 process:
  Thread 1 — CAMERA   : đọc webcam liên tục ~30fps, vẽ khung xương YOLO
                        từng khung + HUD méo mặt MediaPipe → frame phục vụ
  Thread 2 — RADAR    : LD2450 quét 0.6s/nhịp (không chặn camera)
  Thread 3 — PHÂN TÍCH: mỗi 2s face/arm/gait → Defense → Fusion → Alert
                        → NIHSS → Handoff (báo động tức thì vào /api/state)

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
import threading
from collections import deque

import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
for p in (SRC_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, Request
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

# ======================================================================
# LOAD MODEL 1 LẦN
# ======================================================================
print('[WEB] Đang tải model (chờ ~10 giây)...')
from detection.face_module_v7 import FaceAsymmetryDetector      # noqa: E402
from detection.arm_module import ArmWeaknessDetector             # noqa: E402
from detection.gait_module import GaitPoseDetector               # noqa: E402

FACE = FaceAsymmetryDetector(
    model_path=os.path.join(MODELS, 'face_landmarker_v2.task'))
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
}
CAM_LOCK = threading.Lock()
RADAR_LOCK = threading.Lock()
ANALYSIS_LOCK = threading.Lock()

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


# ======================================================================
# VẼ OVERLAY (khung xương YOLO + HUD méo mặt MediaPipe)
# ======================================================================
def draw_live_pose(frame, conf=0.30):
    """Khung xương YOLO vẽ TỪNG KHUNG (tối đa 3 người — camera giám sát)."""
    if ARM.pose_model is None:
        return frame
    try:
        with YOLO_LOCK:
            res = ARM.pose_model.predict(frame, verbose=False, conf=conf,
                                         imgsz=320)
        kpts = res[0].keypoints
        if kpts is None or kpts.xy is None or len(kpts.xy) == 0:
            return frame
        h, w = frame.shape[:2]
        for person in kpts.xy[:3]:
            pts = [(int(x * w), int(y * h))
                   for x, y in person.cpu().numpy()]
            for a, b in YOLO_SKELETON:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], (80, 220, 80), 2,
                             cv2.LINE_AA)
            for p in pts:
                cv2.circle(frame, p, 3, (0, 255, 255), -1, cv2.LINE_AA)
    except Exception:
        pass
    return frame


def draw_face_hud(frame, face_r):
    """HUD đo chính xác: oval hướng dẫn + điểm méo mặt + metric vượt ngưỡng."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, int(h * 0.46)
    axes = (int(w * 0.22), int(h * 0.42))
    status = face_r.get('status', 'NO_FACE')
    lost = status in ('NO_FACE', 'INVALID_LANDMARKS', 'NO_DETECTOR', 'ERROR')
    color = {'DANGER': (0, 0, 255), 'WARNING': (0, 165, 255),
             'NORMAL': (60, 200, 60)}.get(status, (0, 0, 255))
    cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360,
                (0, 0, 255) if lost else (210, 210, 210), 2, cv2.LINE_AA)
    lm = face_r.get('raw_landmarks')
    if lm is not None:
        # chấm landmark MediaPipe (mỗi điểm thứ 2 — 239 chấm) quanh mặt
        for x, y in lm[::2]:
            cv2.circle(frame, (int(x * w), int(y * h)), 1,
                       (230, 200, 150), -1, cv2.LINE_AA)
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
        cv2.putText(frame, 'MAT FACE — dat mat vao khung', (cx - 160, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame


# ======================================================================
# THREAD 1 — CAMERA (~30fps, vẽ overlay trực tiếp)
# ======================================================================
def camera_worker():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        STATE['cam_error'] = 'Không mở được webcam'
        return
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
        # vẽ khung xương YOLO từng khung (real-time)
        frame = draw_live_pose(frame)
        # HUD mặt từ chu kỳ phân tích 2s
        with ANALYSIS_LOCK:
            hud = STATE.get('hud_face')
        if hud:
            frame = draw_face_hud(frame, hud)
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
# THREAD 3 — PHÂN TÍCH (mỗi 2 giây)
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
    STATE['radar'].set_audio_flag(False)   # speech test ở app Streamlit

    ts = time.strftime('%H:%M:%S')
    STATE['radar_history'].append({
        'Thời điểm': ts, 'Trạng thái': radar_r['status'],
        'Dịch chuyển (cm)': radar_r['metrics'].get('position_change_cm', 0.0),
        'Bất động (s)': radar_r['metrics'].get('inactivity_s', 0.0),
        'Chế độ': 'SIM' if STATE['radar'].simulation else 'LIVE'})

    mods = {'face': face_r, 'arm': arm_r, 'gait': gait_r, 'radar': radar_r}
    filtered, audit = DEFENSE.filter(mods)
    fused = FUSION.fuse(filtered)
    verdict = DEFENSE.verdict(fused['fused_score'])
    fused['trend'] = verdict['trend']

    # ----- ALERT -----
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

    # ----- NIHSS + Handoff events -----
    nih = estimate_nihss(filtered)
    nih.update(calculate_nihss_ci(filtered, n_iter=300))
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

    with ANALYSIS_LOCK:
        STATE['hud_face'] = dict(face_r)
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
        }
        STATE['fused_score'] = round(float(fused['fused_score']), 1)
        STATE['risk_level'] = fused['risk_level']
        STATE['verdict'] = verdict
        STATE['nihss'] = nih
        STATE['analysis_count'] += 1


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
      gait:'Dáng đi (YOLO)',radar:'Radar LD2450'};
    for (const k in s.modules){
      const m = s.modules[k];
      mods.innerHTML += `<div class="mod"><span>${name[k]||k}</span>
        <span><span class="st-${esc(m.status)}">${esc(m.status)}</span>
        &nbsp;${m.score}/100</span></div>`;
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
# MAIN
# ======================================================================
def main():
    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=radar_worker, daemon=True).start()
    threading.Thread(target=analysis_worker, daemon=True).start()

    import uvicorn
    print('=' * 64)
    print('  GOLDEN-WATCH WEB — mở trình duyệt:  http://localhost:5001')
    print('  Video MJPEG (mở VLC được): http://localhost:5001/video.mjpg')
    print('  Chỉ nghe trên 127.0.0.1 (localhost) — KHÔNG mở ra mạng ngoài')
    print('  Ctrl+C để dừng · KHÔNG PHẢN CHẨN ĐOÁN — khẩn cấp GỌI 115')
    print('=' * 64)
    uvicorn.run(app, host='127.0.0.1', port=5001, log_level='warning')


if __name__ == '__main__':
    main()
