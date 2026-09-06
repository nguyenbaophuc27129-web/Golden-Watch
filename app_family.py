# -*- coding: utf-8 -*-
"""
GOLDEN-WATCH — APP GIA ĐÌNH (PSCS v8.0)
Ứng dụng Streamlit DÀNH RIÊNG CHO GIA ĐÌNH giám sát người cao tuổi có nguy cơ
đột quỵ: camera + radar giám sát liên tục → kết quả về app → TIN NHẮN BÁO ĐỘNG
trong app để người thân kịp thời cứu trong THỜI GIAN VÀNG (< 4.5 giờ).

5 TAB:
  1. Giám sát        : camera trực tiếp + 5 module + điểm fusion
  2. Báo động        : TIN NHẮN cho người thân (tính năng trung tâm)
  3. Kết quả & NIHSS : NIHSS 4 items ± CI + xu hướng
  4. Bệnh viện       : triage + bệnh viện đề xuất + báo cáo QR/PDF
  5. Kiểm tra nói    : M2 speech (15s → median 3 cửa sổ — M2-10)

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


@st.cache_resource
def get_camera(_key=0):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def init_session():
    ss = st.session_state
    if 'initialized' in ss:
        return
    ss.initialized = True
    ss.monitoring = False
    ss.face, ss.arm, ss.gait = load_detectors()
    ss.fusion = FusionEngine()
    ss.defense = DefenseEngine()
    ss.alerter = AlertSystem(buzzer_enabled=True, cooldown_seconds=60)
    ss.triage = TriageEngine()
    ss.speech_result = None        # từ tab Kiểm tra nói
    ss.last_frame = None
    ss.prev_gray = None
    ss.radar_scenario = 'normal'
    ss.radar = RadarModule(simulation=True, sim_scenario='normal')
    ss.last_fusion = None
    ss.last_nihss = None
    ss.last_verdict = None
    ss.handoff = None
    ss.patient = {'name': '', 'age': '', 'gender': 'Nam', 'phone': '',
                  'address': ''}


# ======================================================================
# CHU KỲ GIÁM SÁT (fragment — chạy mỗi 3 giây, không rerun cả app)
# ======================================================================
@st.fragment(run_every="3s")
def monitor_fragment():
    ss = st.session_state
    if not ss.monitoring:
        return

    cap = get_camera()
    ok, frame = cap.read()
    if not ok or frame is None:
        ss.cam_error = 'Không đọc được webcam'
        return
    ss.cam_error = None
    frame = cv2.flip(frame, 1)

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
    arm_r = ss.arm.detect_arm_weakness(frame)
    gait_r = ss.gait.detect_gait_from_frame(frame)

    # ----- Radar (1 nhịp ngắn) -----
    radar_r = ss.radar.analyze(duration_s=0.6, sample_hz=3.0)
    radar_r['metrics']['simulation'] = ss.radar.simulation
    # AND-gate: audio bất thường nếu speech WARNING/DANGER gần nhất
    sp = ss.speech_result or {}
    ss.radar.set_audio_flag(sp.get('speech_prob', 0) >= 30)

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

    ss.last_frame = frame
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
            if ss.monitoring and ss.handoff is None:
                ss.handoff = HandoffSystem(
                    patient_info=ss.patient)
                ss.handoff.add_event('T1', 'Bat dau giam sat camera+radar')
            st.rerun()
        c2.button('🔄 Làm mới', on_click=st.rerun, use_container_width=True)

        if ss.cam_error:
            st.error(f'Camera lỗi: {ss.cam_error}')
        if ss.last_frame is not None:
            st.image(cv2.cvtColor(ss.last_frame, cv2.COLOR_BGR2RGB),
                     caption='Camera trực tiếp', use_container_width=True)
        else:
            st.info('Nhấn **BẬT giám sát** — camera + radar sẽ quét chu kỳ '
                    '3 giây/lần.')
    with col2:
        st.subheader('📡 Trạng thái 5 module')
        fused = ss.last_fusion
        if fused:
            mu = fused.get('modules_used', {})
            rows = []
            names = {'face': '😀 Méo mặt', 'speech': '🗣️ Nói khó',
                     'arm': '💪 Tay yếu', 'gait': '🚶 Đi bất thường',
                     'radar': '📶 Radar (ngã)'}
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
        else:
            st.info('Chưa có dữ liệu — bật giám sát.')


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
        with st.form('patient'):
            ss.patient['name'] = st.text_input('Họ tên', ss.patient['name'])
            ss.patient['age'] = st.text_input('Tuổi', ss.patient['age'])
            ss.patient['gender'] = st.selectbox(
                'Giới tính', ['Nam', 'Nữ'],
                index=['Nam', 'Nữ'].index(ss.patient['gender']))
            ss.patient['phone'] = st.text_input('Điện thoại', ss.patient['phone'])
            ss.patient['address'] = st.text_input('Địa chỉ', ss.patient['address'])
            st.form_submit_button('💾 Lưu', on_click=lambda: None)
        st.divider()
        st.header('📶 Radar LD2450')
        sim = st.checkbox('SIMULATION (chưa cắm radar)', value=True)
        scenario = st.selectbox('Kịch bản mô phỏng',
                                ['normal', 'fall', 'wander'],
                                disabled=not sim)
        if st.button('Áp dụng radar'):
            ss.radar = RadarModule(
                simulation=sim,
                port=None if sim else 'COM3',
                sim_scenario=scenario)
            st.toast('Đã áp dụng radar')
        st.divider()
        if st.button('🔊 Test còi + tin nhắn EMERGENCY'):
            ss.alerter.send_alert(
                95, 'Test he thong bao dong', level='EMERGENCY')
            st.toast('Đã phát cảnh báo thử')
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
    tabs = st.tabs(['🎬 Giám sát', '🚨 Báo động', '📊 Kết quả & NIHSS',
                    '🏥 Bệnh viện & Báo cáo', '🗣️ Kiểm tra nói'])
    with tabs[0]:
        monitor_fragment()
        tab_monitor()
    with tabs[1]:
        tab_alerts()
    with tabs[2]:
        tab_results()
    with tabs[3]:
        tab_hospital()
    with tabs[4]:
        tab_speech()


if __name__ == '__main__':
    main()
