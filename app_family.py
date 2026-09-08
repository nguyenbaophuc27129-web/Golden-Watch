# -*- coding: utf-8 -*-
"""
GOLDEN-WATCH — APP GIA ĐÌNH (PSCS v8.0)
Ứng dụng Streamlit DÀNH RIÊNG CHO GIA ĐÌNH giám sát người cao tuổi có nguy cơ
đột quỵ: camera + radar giám sát liên tục → kết quả về app → TIN NHẮN BÁO ĐỘNG
trong app để người thân kịp thời cứu trong THỜI GIAN VÀNG (< 4.5 giờ).

APP 2 PHẦN (v8.1):
  📷 PHẦN 1 — CAMERA  : giám sát camera (méo mặt / tay yếu / dáng đi)
                        + kiểm tra nói (M2)
  📡 PHẦN 2 — RADAR   : LD2450 (ngã / bất động) — chọn COM port thật
                        hoặc SIMULATION, kịch bản mô phỏng
  CHUNG               : Báo động (tính năng trung tâm) · Kết quả & NIHSS
                        · Bệnh viện & Báo cáo QR/PDF

Ghi chú trung thực (M4-07): model gait ML (PhysioNet force-plate) KHÔNG nối
vào đường camera vì sai khác không gian đặc trưng (8 đặc trưng cảm biến lực
vs 6 chỉ số pose) — camera gait dùng luật; model v2 dành cho cảm biến lực.

Chạy:
    venv/Scripts/streamlit run app_family.py
(cần webcam; radar chạy SIMULATION nếu chưa cắm LD2450)

KHÔNG PHẢN CHẨN ĐOÁN — công cụ hỗ trợ phát hiện sớm. Khẩn cấp: GỌI 115.

Tác giả: PSCS Team
Ngày: 06/09/2026
"""

import os
import sys
import time
import threading

import numpy as np
import streamlit as st
import cv2

# ---- Path setup ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
for p in (SRC_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from fusion.fusion_engine import FusionEngine
from fusion.nihss_estimator import estimate_nihss, calculate_nihss_ci
from fusion.triage_engine import TriageEngine
from defense.defense_engine import DefenseEngine
from alerts.alert_system import AlertSystem
from handoff.handoff_system import HandoffSystem
from detection.radar_module import RadarModule

MODELS = os.path.join(PROJECT_ROOT, 'models')

# ======================================================================
# TRẠNG THÁI (init 1 lần)
# ======================================================================
st.set_page_config(page_title="Golden-Watch — Bảo vệ thời gian vàng",
                   page_icon="⏱️", layout="wide")

RISK_COLOR = {'NORMAL': '#1a7a3a', 'MONITOR': '#b8860b',
              'WARNING': '#d97706', 'EMERGENCY': '#c81e1e'}


@st.cache_resource(show_spinner=False)
def load_detectors():
    """Load 1 lần: face (MediaPipe) + arm/gait (YOLO pose + ML)."""
    from detection.face_module_v7 import FaceAsymmetryDetector
    from detection.arm_module import ArmWeaknessDetector
    from detection.gait_module import GaitPoseDetector

    face = FaceAsymmetryDetector(
        model_path=os.path.join(MODELS, 'face_landmarker_v2.task'))
    arm = ArmWeaknessDetector(
        pose_model_path=os.path.join(SRC_DIR, 'yolov8n-pose.pt'),
        ml_model_path=os.path.join(
            MODELS, 'arm_weakness_20260830_200657.pth'),
        scaler_path=os.path.join(
            MODELS, 'arm_weakness_20260830_200657_scaler.pkl'))
    gait = GaitPoseDetector(
        model_path=os.path.join(SRC_DIR, 'yolov8n-pose.pt'))
    return face, arm, gait


# ======================================================================
# CAMERA GIÁM SÁT THỜI GIAN THỰC — thread đọc liên tục (như CCTV)
# ======================================================================
YOLO_SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4),          # đầu
                 (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # tay
                 (5, 11), (6, 12), (11, 12),               # thân
                 (11, 13), (13, 15), (12, 14), (14, 16)]   # chân


def _camera_worker(state):
    """Thread: đọc webcam LIÊN TỤC (~30fps) vào buffer — fragment chỉ lấy
    frame mới nhất để hiển thị, không bao giờ chờ phân tích."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    state['cap'] = cap
    if not cap.isOpened():
        state['error'] = 'Không mở được webcam'
        return
    while not state['stop']:
        ok, frame = cap.read()
        if not ok or frame is None:
            state['error'] = 'Không đọc được webcam'
            time.sleep(0.2)
            continue
        state['error'] = None
        frame = cv2.flip(frame, 1)          # gương — tự nhiên khi xem
        with state['lock']:
            state['frame'] = frame
        time.sleep(0.03)
    cap.release()


def _radar_worker(state):
    """Thread riêng cho radar — video KHÔNG bị chặn bởi quét 0.6s."""
    while not state['stop']:
        radar = state.get('radar')
        try:
            state['result'] = (radar.analyze(duration_s=0.6, sample_hz=3.0)
                               if radar else None)
        except Exception as e:
            state['result'] = {'status': 'ERROR', 'metrics': {'err': str(e)}}
        time.sleep(1.4)


def start_workers():
    """Bật 2 thread nền: camera + radar (daemon — thoát app tự tắt)."""
    ss = st.session_state

    def alive(c):
        return c is not None and c.get('thread') is not None \
            and c['thread'].is_alive()

    if not alive(ss.cam_state):
        cs = {'stop': False, 'lock': threading.Lock(),
              'frame': None, 'error': None}
        t = threading.Thread(target=_camera_worker, args=(cs,), daemon=True)
        cs['thread'] = t
        ss.cam_state = cs
        t.start()
    if not alive(ss.radar_state):
        rs = {'stop': False, 'result': None, 'radar': ss.radar}
        t = threading.Thread(target=_radar_worker, args=(rs,), daemon=True)
        rs['thread'] = t
        ss.radar_state = rs
        t.start()
    else:
        ss.radar_state['radar'] = ss.radar   # radar đổi trong tab 📡


def stop_workers():
    ss = st.session_state
    for state in (ss.cam_state, ss.radar_state):
        if state is not None:
            state['stop'] = True
    ss.cam_state = None
    ss.radar_state = None
    ss.hud_face = None
    ss.prev_gray = None


def draw_live_pose(frame, pose_model, lock, conf=0.30):
    """Vẽ khung xương YOLO TRỰC TIẾP mỗi khung video (tối đa 3 người).
    Chỉ VIZ — không ảnh hưởng phân tích (phân tích chạy chu kỳ riêng)."""
    if pose_model is None:
        return frame
    try:
        with lock:
            res = pose_model.predict(frame, verbose=False, conf=conf,
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


def init_session():
    ss = st.session_state
    if 'initialized' in ss:
        return
    ss.initialized = True
    ss.monitoring = False
    ss.cam_error = None
    ss.face, ss.arm, ss.gait = load_detectors()
    ss.fusion = FusionEngine()
    ss.defense = DefenseEngine()
    ss.alerter = AlertSystem(buzzer_enabled=True, cooldown_seconds=60)
    ss.triage = TriageEngine()
    ss.speech_result = None        # từ tab Kiểm tra nói
    from detection.face_ml_5feat import FaceML5Feat
    ss.face_ml = FaceML5Feat.load_latest(MODELS)   # M1-07 (mặc định TẮT)
    ss.face_ml_enabled = False                     # bật sau protocol B (SYS-15)
    ss.prev_gray = None
    ss.hud_face = None        # kết quả face gần nhất cho HUD video
    ss.cam_state = None       # {frame, lock, stop, thread} — thread camera
    ss.radar_state = None     # {result, radar, stop, thread} — thread radar
    ss.yolo_lock = threading.Lock()   # 1 YOLO model, 2 thread dùng
    ss.radar_scenario = 'normal'
    ss.radar = RadarModule(simulation=True, sim_scenario='normal')
    ss.radar_history = []
    ss.last_fusion = None
    ss.last_nihss = None
    ss.last_verdict = None
    ss.handoff = None
    ss.patient = {'name': '', 'age': '', 'gender': 'Nam', 'phone': '',
                  'address': ''}


# ======================================================================
# KHUNG ĐO CHÍNH XÁC (HUD) — vẽ lên frame hiển thị, SAU khi module đã chạy
# ======================================================================
FACE_THRESHOLDS = {'mouth_ratio': (0.25, '%'), 'eye_ratio': (0.30, '%'),
                   'face_tilt': (10.0, 'do'), 'nasolabial_ratio': (0.35, '%'),
                   'forehead_ratio': (0.30, '%')}
FACE_NAMES = {'mouth_ratio': 'meo mieng', 'eye_ratio': 'lech nhan',
              'face_tilt': 'nghieng dau', 'nasolabial_ratio': 'ranh mui-moi',
              'forehead_ratio': 'nep tran'}


def draw_face_hud(frame, face_r):
    """Vẽ oval hướng dẫn đặt mặt + điểm méo mặt live + metric vượt ngưỡng.
    Ghi chú: cv2.putText chỉ hiển thị ASCII — dùng chữ không dấu."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, int(h * 0.46)
    axes = (int(w * 0.22), int(h * 0.42))
    status = face_r.get('status', 'NO_FACE')
    lost = status in ('NO_FACE', 'INVALID_LANDMARKS', 'NO_DETECTOR', 'ERROR')
    color = {'DANGER': (0, 0, 255), 'WARNING': (0, 165, 255),
             'NORMAL': (60, 200, 60)}.get(status, (0, 0, 255))
    # Oval hướng dẫn — đặt mặt vào đây (mắt ngang vạch giữa)
    cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360,
                (0, 0, 255) if lost else (210, 210, 210), 2, cv2.LINE_AA)

    # Hộp quanh mặt thật (từ 478 landmarks) + chấm landmark MediaPipe
    lm = face_r.get('raw_landmarks')
    if lm is not None:
        for x, y in lm[::2]:
            cv2.circle(frame, (int(x * w), int(y * h)), 1,
                       (230, 200, 150), -1, cv2.LINE_AA)
        xs, ys = lm[:, 0], lm[:, 1]
        cv2.rectangle(frame, (int(xs.min() * w), int(ys.min() * h)),
                      (int(xs.max() * w), int(ys.max() * h)), color, 2)

    # Thanh điểm méo mặt + vạch ngưỡng 30
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

    # Metric vượt ngưỡng (tối đa 3 dòng) + góc nghiêng hiện tại
    rm = face_r.get('raw_metrics') or {}
    y = 26
    tilt = rm.get('face_tilt')
    if tilt is not None:
        cv2.putText(frame,
                    f'nghieng dau: {tilt:.0f} do (WARNING khi ~>=25)',
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
# CHU KỲ PHÂN TÍCH (fragment — mỗi 2 giây; VIDEO hiển thị ở video_fragment)
# ======================================================================
@st.fragment(run_every="2s")
def monitor_fragment():
    ss = st.session_state
    if not ss.monitoring or ss.cam_state is None:
        return
    if ss.cam_state.get('error'):
        ss.cam_error = ss.cam_state['error']
        return
    with ss.cam_state['lock']:
        frame = (None if ss.cam_state.get('frame') is None
                 else ss.cam_state['frame'].copy())
    if frame is None:
        return
    ss.cam_error = None

    # ----- Cường độ chuyển động (L2 Context) -----
    gray = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
    motion = 0.0
    if ss.prev_gray is not None:
        motion = float(np.mean(np.abs(
            gray.astype(np.int16) - ss.prev_gray.astype(np.int16)))) / 50.0
        motion = min(motion, 1.0)
    ss.prev_gray = gray

    # ----- 3 module từ camera -----
    face_r = ss.face.process_frame(frame)
    # M1-07: face ML 5-feat — CHỈ khi bật flag (thử nghiệm, chờ protocol B)
    if ss.face_ml_enabled and ss.face_ml is not None:
        ml_p = ss.face_ml.predict(face_r.get('raw_metrics'))
        if ml_p is not None:
            face_r['metrics']['rules_prob'] = face_r.get('score')
            face_r['score'] = round(ml_p, 1)
            face_r['status'] = ('NORMAL' if ml_p < 30 else
                                'WARNING' if ml_p < 60 else 'DANGER')
    arm_r = ss.arm.detect_arm_weakness(frame)
    gait_r = ss.gait.detect_gait_from_frame(frame)

    # ----- Radar (từ thread riêng — không chặn) -----
    radar_r = (ss.radar_state or {}).get('result') or {
        'status': 'NO_DATA', 'metrics': {}}
    radar_r.setdefault('metrics', {})['simulation'] = ss.radar.simulation
    # AND-gate: audio bất thường nếu speech WARNING/DANGER gần nhất
    sp = ss.speech_result or {}
    ss.radar.set_audio_flag(sp.get('speech_prob', 0) >= 30)
    # Lịch radar cho tab 📡 Radar
    ss.radar_history.append({
        'Thời điểm': time.strftime('%H:%M:%S'),
        'Trạng thái': radar_r['status'],
        'Dịch chuyển (cm)': radar_r['metrics'].get('position_change_cm', 0.0),
        'Bất động (s)': radar_r['metrics'].get('inactivity_s', 0.0),
        'Chế độ': 'SIM' if ss.radar.simulation else 'LIVE',
    })
    if len(ss.radar_history) > 120:
        ss.radar_history = ss.radar_history[-120:]

    mods = {'face': face_r, 'arm': arm_r, 'gait': gait_r, 'radar': radar_r}
    if ss.speech_result:
        mods['speech'] = ss.speech_result

    # ----- Defense (L2 context + L4) → Fusion → Defense L3 verdict -----
    ss.defense.update_context(motion_intensity=motion,
                              talking=(face_r.get('status') != 'NO_FACE'
                                       and motion > 0.02))
    filtered, audit = ss.defense.filter(mods)
    fused = ss.fusion.fuse(filtered)
    verdict = ss.defense.verdict(fused['fused_score'])
    fused['trend'] = verdict['trend']

    # ----- ALERT (buzzer + log + app feed) -----
    alert_res = None
    if verdict['alert']:
        fused_for_alert = dict(fused)
        fused_for_alert['recommendation'] = (
            'GOI 115 — khong de nhan benh an/uong nuoc')
        alert_res = ss.alerter.process_fusion_result(fused_for_alert)
        if alert_res.get('triggered'):
            st.toast('🚨 CẢNH BÁO ĐỘT QUỴ — xem tab BÁO ĐỘNG!', icon='🚨')

    # ----- NIHSS ước tính -----
    nih = estimate_nihss(filtered)
    ci = calculate_nihss_ci(filtered, n_iter=300)
    nih.update(ci)

    ss.hud_face = dict(face_r)   # HUD (landmark + điểm) cho video trực tiếp
    ss.last_fusion = fused
    ss.last_nihss = nih
    ss.last_verdict = verdict
    ss.last_module_counts = {
        'face': face_r.get('status', '?'), 'arm': arm_r.get('status', '?'),
        'gait': gait_r.get('status', '?'), 'radar': radar_r['status'],
    }
    ss.trend_points = getattr(ss, 'trend_points', []) + [
        {'Thời điểm': time.strftime('%H:%M:%S'),
         'Điểm nguy cơ': fused['fused_score']}]
    if len(ss.trend_points) > 120:
        ss.trend_points = ss.trend_points[-120:]


# ======================================================================
# VIDEO TRỰC TIẾP (fragment ~7 khung/giây — như camera giám sát)
# ======================================================================
@st.fragment(run_every="0.15s")
def video_fragment():
    ss = st.session_state
    if not ss.monitoring or ss.cam_state is None:
        st.info('Nhấn **BẬT giám sát** — camera trực tiếp liên tục '
                '(khung xương YOLO + khung đo méo mặt MediaPipe).')
        return
    with ss.cam_state['lock']:
        frame = (None if ss.cam_state.get('frame') is None
                 else ss.cam_state['frame'].copy())
    if frame is None:
        st.warning('Đang kết nối camera...')
        return
    if ss.cam_state.get('error'):
        st.error(f"Camera lỗi: {ss.cam_state['error']}")
        return

    # 1) Khung xương YOLO — VẼ TRỰC TIẾP từng khung (chuyển động real-time)
    frame = draw_live_pose(frame, ss.arm.pose_model, ss.yolo_lock)
    # 2) MediaPipe méo mặt + HUD đo chính xác — từ chu kỳ phân tích 2s
    if ss.hud_face:
        frame = draw_face_hud(frame, ss.hud_face)

    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
             use_container_width=True)
    mode = 'LIVE' if not ss.radar.simulation else 'SIM radar'
    st.caption(f'🔴 TRỰC TIẾP {time.strftime("%H:%M:%S")} · khung xương YOLO '
               f'từng khung · méo mặt (MediaPipe) & điểm cập nhật mỗi 2s '
               f'· radar {mode}')


# ======================================================================
# UI CHÍNH
# ======================================================================
def header():
    ss = st.session_state
    risk = (ss.last_fusion or {}).get('risk_level', '—')
    score = (ss.last_fusion or {}).get('fused_score', 0)
    c1, c2, c3 = st.columns([2.2, 1, 1])
    c1.title('⏱️ Golden-Watch')
    c1.caption('Bảo vệ người thân trong thời gian vàng — PSCS v8.0')
    c2.metric('Điểm nguy cơ', f'{score:.0f}/100')
    c3.markdown(
        f"<div style='text-align:center'>"
        f"<span style='background:{RISK_COLOR.get(risk,'#555')};color:white;"
        f"padding:10px 18px;border-radius:12px;font-size:22px;font-weight:bold'>"
        f"{risk}</span></div>", unsafe_allow_html=True)
    if not ss.patient.get('name'):
        st.warning('👉 Nhập **thông tin người bệnh** ở thanh bên để báo cáo '
                   'bàn giao đầy đủ. Khẩn cấp: **GỌI 115**.')


def tab_monitor():
    ss = st.session_state
    col1, col2 = st.columns([1.4, 1])
    with col1:
        c1, c2 = st.columns([1, 1])
        if c1.button('▶️ BẬT giám sát' if not ss.monitoring else '⏸️ Dừng',
                     type='primary', use_container_width=True):
            ss.monitoring = not ss.monitoring
            if ss.monitoring:
                start_workers()
                if ss.handoff is None:
                    ss.handoff = HandoffSystem(
                        patient_info=ss.patient)
                    ss.handoff.add_event('T1', 'Bat dau giam sat camera+radar')
            else:
                stop_workers()
            st.rerun()
        c2.button('🔄 Làm mới', on_click=st.rerun, use_container_width=True)

        if ss.cam_error:
            st.error(f'Camera lỗi: {ss.cam_error}')
        video_fragment()
    with col2:
        st.subheader('📷 Trạng thái module camera')
        fused = ss.last_fusion
        if fused:
            mu = fused.get('modules_used', {})
            rows = []
            names = {'face': '😀 Méo mặt', 'speech': '🗣️ Nói khó',
                     'arm': '💪 Tay yếu', 'gait': '🚶 Đi bất thường'}
            for k, label in names.items():
                if k in mu:
                    rows.append(f"{label}: **{mu[k]['prob']:.0f}%** — "
                                f"{mu[k]['status']}")
                else:
                    rows.append(f"{label}: ⚪ không có dữ liệu")
            st.markdown('\n'.join(rows))
            st.progress(min(fused['fused_score'], 100) / 100,
                        text=f"Fusion: {fused['fused_score']:.0f}/100 — "
                             f"{fused['risk_level']}")
            if fused.get('triggered_rules'):
                st.caption('Luật FAST: ' + '; '.join(fused['triggered_rules']))
            if ss.last_verdict:
                st.caption(f"Defense L3: {ss.last_verdict['reason']}")
            st.caption(f"Defense: {ss.defense.get_stats()}")
            st.caption('📶 Radar (ngã/bất động): xem tab **📡 Radar** — '
                       f"hiện tại: **{ss.radar_history[-1]['Trạng thái']}** "
                       f"({ss.radar_history[-1]['Chế độ']})"
                       if ss.radar_history else
                       '📶 Radar: chưa có dữ liệu — xem tab **📡 Radar**')
        else:
            st.info('Chưa có dữ liệu — bật giám sát.')


def tab_radar():
    """📡 PHẦN 2 — RADAR LD2450: ngã / bất động, port thật hoặc mô phỏng."""
    ss = st.session_state
    st.subheader('📡 Radar LD2450 — phát hiện ngã & bất động')
    st.caption('Ngã: dịch chuyển đột biến > 100 cm HOẶC bất động > 45 s sau '
               'biến động vị trí. Âm thanh bất thường (speech ≥ 30%) sẽ '
               'củng cố cảnh báo (AND-gate ×0.3 ngưỡng).')

    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown('**Cấu hình kết nối**')
        sim = st.checkbox('SIMULATION (chưa cắm radar)', value=ss.radar.simulation)
        ports = RadarModule.list_ports()
        port_labels = [f"{dev} — {desc}" for dev, desc in ports]
        port_sel = st.selectbox(
            'COM port (radar thật @ 256000 baud)',
            port_labels if port_labels
            else ['(không thấy COM port — cắm LD2450)'],
            disabled=sim)
        scenario = st.selectbox('Kịch bản mô phỏng',
                                ['normal', 'fall', 'wander'],
                                disabled=not sim)
        if st.button('Áp dụng radar', type='primary', use_container_width=True):
            # đóng serial radar cũ trước khi thay (tránh kẹt COM port)
            if getattr(ss.radar, 'serial', None) is not None:
                try:
                    ss.radar.serial.close()
                except Exception:
                    pass
            device = port_sel.split(' — ')[0] if port_labels else None
            ok_port = bool(device and device.startswith('COM'))
            ss.radar = RadarModule(
                simulation=sim,
                port=None if (sim or not ok_port) else device,
                sim_scenario=scenario)
            # radar đổi → thread radar dùng module mới ngay nhịp kế tiếp
            if ss.radar_state is not None:
                ss.radar_state['radar'] = ss.radar
                ss.radar_state['result'] = None
            ss.radar_history = []
            ss.radar_mode = ('🟡 SIMULATION' if sim else
                             f'🟢 LIVE @ {device}' if ok_port
                             else '🟡 SIMULATION (port lỗi)')
            st.toast(f'Đã áp dụng radar — {ss.radar_mode}')
        if 'radar_mode' not in ss:
            ss.radar_mode = ('🟡 SIMULATION' if ss.radar.simulation
                             else '🟢 LIVE')
        st.metric('Chế độ', ss.radar_mode,
                  delta=(f"scenario: {ss.radar.sim_scenario}"
                         if ss.radar.simulation else 'LD2450 256000 baud'))
        if st.button('🔊 Test còi + tin nhắn EMERGENCY'):
            ss.alerter.send_alert(
                95, 'Test he thong bao dong', level='EMERGENCY')
            st.toast('Đã phát cảnh báo thử')

    with c2:
        st.markdown('**Trạng thái hiện tại**')
        if ss.radar_history:
            last = ss.radar_history[-1]
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric('Trạng thái', last['Trạng thái'])
            sc2.metric('Dịch chuyển', f"{last['Dịch chuyển (cm)']:.0f} cm",
                       delta='ngã nếu > 100')
            sc3.metric('Bất động', f"{last['Bất động (s)']:.0f} s",
                       delta='ngã nếu > 45')
            import pandas as pd
            df = pd.DataFrame(ss.radar_history)
            st.line_chart(df.set_index('Thời điểm')[
                ['Dịch chuyển (cm)', 'Bất động (s)']])
            st.caption('120 nhịp gần nhất (mỗi 3 s). Cột "Chế độ": '
                       + ' / '.join(sorted(df['Chế độ'].unique())))
        else:
            st.info('Chưa có dữ liệu radar — bật **BẬT giám sát** ở 📷 '
                    'Giám sát camera (radar chạy cùng chu kỳ 3 s).')


def tab_alerts():
    """TÍNH NĂNG TRUNG TÂM: tin nhắn báo động cho người thân."""
    ss = st.session_state
    st.subheader('🚨 Tin nhắn báo động gửi người thân')
    st.caption('Mỗi cảnh báo kèm việc cần làm NGAY. Thời gian vàng từ khi có '
               'dấu hiệu: < 4,5 giờ — mỗi phút chậm trễ mất ~1,9 triệu neuron '
               '(Saver 2006).')
    feed = ss.alerter.get_recent_alerts(20)
    if not feed:
        st.success('✅ Chưa có cảnh báo nào. Người thân đang an toàn '
                   '(theo dõi sẽ hiện tại đây khi có dấu hiệu).')
        return
    for a in feed:
        if a['level'] == 'EMERGENCY':
            st.error(f"**🔴 {a['timestamp']} — ĐỘT QUỴ NGUY CƠ CAO "
                     f"({a['score']:.0f}/100)**\n\n{a['text']}")
        elif a['level'] == 'WARNING':
            st.warning(f"**🟡 {a['timestamp']} — Dấu hiệu bất thường "
                       f"({a['score']:.0f}/100)**\n\n{a['text']}")
        else:
            st.info(f"ℹ️ {a['timestamp']} — {a['text']}")


def tab_results():
    ss = st.session_state
    st.subheader('📊 Kết quả & ước tính NIHSS')
    if not ss.last_fusion:
        st.info('Bật giám sát ở tab Giám sát để có kết quả.')
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### Điểm nguy cơ: **{ss.last_fusion['fused_score']:.0f}"
                    f"/100 — {ss.last_fusion['risk_level']}**")
        st.markdown(f"Khuyến nghị: **{ss.last_fusion['recommendation']}**")
        if ss.trend_points:
            import pandas as pd
            st.line_chart(pd.DataFrame(ss.trend_points).set_index('Thời điểm'))
            st.caption('Xu hướng điểm nguy cơ (120 điểm gần nhất)')
    with c2:
        nih = ss.last_nihss
        st.markdown(f"### NIHSS ước tính: **{nih['total']} ± {nih['margin']}** "
                    f"(CI95% {nih['ci_low']}–{nih['ci_high']}, tối đa 13)")
        vn = {'item4_facial_palsy': 'Méo mặt (Item 4)',
              'item5_motor_arm': 'Tay yếu (Item 5)',
              'item6_motor_leg': 'Chân yếu (Item 6)',
              'item10_dysarthria': 'Nói khó (Item 10)'}
        for it, label in vn.items():
            v = nih['items'].get(it)
            st.markdown(f"- {label}: **{v if v is not None else 'không đo được'}**")
        st.caption('⚠️ Ước tính hỗ trợ — bác sĩ chấm NIHSS chuẩn trên lâm sàng. '
                   'KHÔNG PHẢN CHẨN ĐOÁN.')


def tab_hospital():
    ss = st.session_state
    st.subheader('🏥 Sơ bộ & Bệnh viện đề xuất')
    st.caption('KHÔNG PHẢN CHẨN ĐOÁN — gợi ý sơ bộ để trao đổi với bác sĩ.')
    if not ss.last_fusion:
        st.info('Bật giám sát trước.')
        return
    symptoms = {
        'sudden_severe_headache': st.checkbox('Đau đầu dữ dội đột ngột'),
        'vomiting': st.checkbox('Nôn'),
        'gradual_progression': st.checkbox('Tiến triển từ từ'),
    }
    tri = ss.triage.triage(ss.last_fusion, ss.last_nihss['total'],
                           symptoms=symptoms,
                           module_results=ss.last_fusion.get('modules_used', {}))
    c1, c2 = st.columns(2)
    with c1:
        sub = tri['subtype']
        st.markdown(f"**Nghi:** {sub['subtype'].upper()} "
                    f"({sub['confidence']}%) — Mức độ: **{tri['severity']}**")
        if sub['reasons']:
            st.caption('; '.join(sub['reasons']))
        st.info(tri['disclaimer'])
    with c2:
        hosp = tri.get('hospital')
        if hosp:
            st.success(f"**{hosp['name']}**\n\n{hosp['address']}\n\n"
                       f"☎ {hosp['phone']} — {hosp.get('notes','')}")
        else:
            st.warning('Chưa có cơ sở dữ liệu bệnh viện — GỌI 115.')

    st.divider()
    st.markdown('### 📋 Báo cáo bàn giao (Handoff) — QR + PDF')
    if st.button('🧾 Tạo báo cáo QR + PDF'):
        ss.handoff = ss.handoff or HandoffSystem(patient_info=ss.patient)
        verdict = ss.last_verdict or {'reason': 'manual'}
        ss.handoff.add_event('T3', f"Canh bao {ss.last_fusion['risk_level']} "
                             f"(score {ss.last_fusion['fused_score']})")
        nih = ss.last_nihss
        report = ss.handoff.build_report(ss.last_fusion, nih, tri,
                                         alerts=ss.alerter.get_recent_alerts(10),
                                         defense_stats=ss.defense.get_stats())
        qr_path, qr_data = ss.handoff.generate_qr(report)
        report['qr_path'] = qr_path
        pdf_path = ss.handoff.generate_pdf(report)
        json_path = ss.handoff.save_report(report)
        ss.last_report = {'qr_path': qr_path, 'pdf_path': pdf_path,
                          'json_path': json_path}
    if getattr(ss, 'last_report', None):
        c1, c2, c3 = st.columns(3)
        c1.image(ss.last_report['qr_path'], caption='QR tóm tắt cho y tế',
                 width=220)
        with open(ss.last_report['pdf_path'], 'rb') as f:
            c2.download_button('⬇️ Tải PDF', f, 'pscs_handoff.pdf',
                               'application/pdf', use_container_width=True)
        with open(ss.last_report['json_path'], 'rb') as f:
            c3.download_button('⬇️ Tải JSON', f, 'pscs_handoff.json',
                               'application/json', use_container_width=True)


def tab_speech():
    """M2 speech — 15 giây → median 3 cửa sổ (M2-10 vào production)."""
    ss = st.session_state
    st.subheader('🗣️ Kiểm tra nói (15 giây)')
    st.caption('Người bệnh đọc to 1 câu quen thuộc (VD: "Một hai ba bốn năm '
               'sáu bảy tám chín mười"). Hệ thống phân tích 3 cửa sổ 5s có '
               'tiếng nhất và lấy MEDIAN — chống nhiễu (M2-10).')
    if st.button('🎤 Ghi âm 15 giây', type='primary'):
        try:
            import sounddevice as sd
            from detection.speech_module_v2 import SpeechAnalysisModule

            if 'speech_module' not in ss:
                ss.speech_module = SpeechAnalysisModule(
                    vosk_model_path=os.path.join(MODELS, 'vosk-model-vn-0.4'),
                    ml_model_path=os.path.join(
                        MODELS, 'speech_torgo_20260828_211130.pth'),
                    scaler_path=os.path.join(
                        MODELS, 'speech_torgo_20260828_211130_scaler.pkl'))
                ss.speech_module.load_baseline(
                    os.path.join(PROJECT_ROOT, 'data', 'baselines',
                                 'user_default.json'))
            with st.status('Đang ghi 15 giây... HÃY NÓI LIÊN TỤC', expanded=True):
                audio = ss.speech_module.record_audio(duration_seconds=15)
            if audio is not None:
                # M2-10: 3 cửa sổ có tiếng nhất → median
                probs, ratios = [], []
                with st.spinner('Phân tích 3 cửa sổ...'):
                    for k in range(3):
                        win, info = ss.speech_module.extract_window(audio, 5)
                        r = ss.speech_module.predict_dysarthria(win, 5)
                        probs.append(r['speech_prob'])
                        ratios.append(info['window_speech_ratio'])
                median_prob = sorted(probs)[1]
                sp_status = ('NORMAL' if median_prob < 30 else
                             'WARNING' if median_prob < 60 else 'DANGER')
                ss.speech_result = {
                    'speech_prob': median_prob, 'status': sp_status,
                    'nihss_score': (0 if median_prob < 30 else
                                    1 if median_prob < 60 else 2),
                    'metrics': {'window_probs': probs,
                                'window_ratios': ratios}}
                st.metric('Nói khó (median 3 cửa sổ)', f'{median_prob:.0f}%')
                st.caption(f'Các cửa sổ: '
                           + ', '.join(f'{p:.0f}%' for p in probs))
                st.success(f'Kết quả: {sp_status} — đã đưa vào fusion.')
        except Exception as e:
            st.error(f'Lỗi ghi âm/phân tích: {e}')
    if ss.speech_result:
        st.info(f"Kết quả hiện tại: {ss.speech_result['speech_prob']:.0f}% "
                f"({ss.speech_result['status']})")


# ======================================================================
# SIDEBAR
# ======================================================================
def sidebar():
    ss = st.session_state
    with st.sidebar:
        st.header('👤 Người bệnh')
        with st.form('patient_form'):
            ss.patient['name'] = st.text_input('Họ tên', ss.patient['name'])
            ss.patient['age'] = st.text_input('Tuổi', ss.patient['age'])
            ss.patient['gender'] = st.selectbox(
                'Giới tính', ['Nam', 'Nữ'],
                index=['Nam', 'Nữ'].index(ss.patient['gender']))
            ss.patient['phone'] = st.text_input('Điện thoại', ss.patient['phone'])
            ss.patient['address'] = st.text_input('Địa chỉ', ss.patient['address'])
            st.form_submit_button('💾 Lưu', on_click=lambda: None)
        st.divider()
        st.caption('📶 Cấu hình radar LD2450 đã chuyển sang '
                   'tab **📡 Radar** (chọn COM port thật hoặc mô phỏng).')
        st.divider()
        st.markdown('**Thử nghiệm (M1-07)**')
        ml_ok = ss.face_ml is not None
        ss.face_ml_enabled = st.checkbox(
            'Face ML 5-feat (thay prob rules)',
            value=ss.face_ml_enabled, disabled=not ml_ok,
            help='Model Logistic AUC 0.845 (block-CV) — CHƯA qua protocol B, '
                 'chỉ dùng để so sánh tại chỗ.')
        st.caption(f"Artifact: {'sẵn sàng' if ml_ok else 'không có'}"
                   + (f" (AUC OOF {ss.face_ml.auc_oof})"
                      if ml_ok and ss.face_ml.auc_oof else ''))
        st.divider()
        st.caption('⚠️ **KHÔNG PHẢN CHẨN ĐOÁN.** Công cụ hỗ trợ phát hiện '
                   'sớm đột quỵ. Khẩn cấp: **GỌI 115**.')


# ======================================================================
# MAIN
# ======================================================================
def main():
    init_session()
    sidebar()
    header()

    st.markdown('## 📷 PHẦN 1 — CAMERA (méo mặt · tay yếu · dáng đi · nói)')
    tabs_cam = st.tabs(['🎬 Giám sát camera', '🗣️ Kiểm tra nói'])
    with tabs_cam[0]:
        monitor_fragment()
        tab_monitor()
    with tabs_cam[1]:
        tab_speech()

    st.divider()
    st.markdown('## 📡 PHẦN 2 — RADAR LD2450 (ngã · bất động)')
    tabs_radar = st.tabs(['📡 Giám sát radar'])
    with tabs_radar[0]:
        tab_radar()

    st.divider()
    st.markdown('## 🧩 KẾT QUẢ HỢP NHẤP (camera + radar → fusion)')
    tabs_out = st.tabs(['🚨 Báo động', '📊 Kết quả & NIHSS',
                        '🏥 Bệnh viện & Báo cáo'])
    with tabs_out[0]:
        tab_alerts()
    with tabs_out[1]:
        tab_results()
    with tabs_out[2]:
        tab_hospital()


if __name__ == '__main__':
    main()
