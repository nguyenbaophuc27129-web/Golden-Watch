# -*- coding: utf-8 -*-
"""
TEST HỒI QUY PRE-FLIGHT (SYS-14b) — chạy TRƯỚC mỗi buổi demo/nghiệm thu.

Chặn lại các sự cố đã từng xảy ra:
  SYS-14: file .task bị stub 9KB → face detector "numpy-based (no detection)"
          mà app vẫn chạy (lỗi câm).
  08/09 : arm module crash 'index 2 out of bounds' khi khung hình CÓ người
          (keypoints.xy trả (17,2) thiếu trục confidence — ĐÃ SỬA, test chặn lại).
  08/09 : numpy 2.5 làm numba/librosa gãy → speech mất pitch features.

Chạy:  python src/test_sys14b_regression.py   → exit 0 = PASS, 1 = FAIL
"""

import os
import sys
import glob
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
MODELS = os.path.join(parent_dir, 'models')
SRC = os.path.join(parent_dir, 'src')

FACE_DS = (r"C:\Users\Admin\Documents\NCKHKT_26\fga_project\data"
           r"\datasets\face\Annotated stroke and non stroke Dataset")

results = []


def check(name, ok, detail=''):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" +
          (f" — {detail}" if detail else ''))


def main():
    print('===== SYS-14b PRE-FLIGHT REGRESSION =====')

    # 1. File .task thật (chặn stub 9KB)
    print('[1] Face .task file (SYS-14):')
    cands = (glob.glob(os.path.join(MODELS, 'face_landmarker_v2', '*.task')) +
             glob.glob(os.path.join(MODELS, '*.task')))
    check('có ít nhất 1 file .task', len(cands) > 0)
    big = [p for p in cands if os.path.getsize(p) > 1_000_000]
    check('file .task > 1MB (không phải stub)', len(big) > 0,
          ', '.join(f'{os.path.basename(p)} '
                    f'({os.path.getsize(p)//1024}KB)' for p in cands) or '-')

    # 2. Face detector trên ảnh mặt thật → không được NO_DETECTOR
    print('[2] Face detector ảnh mặt thật:')
    try:
        import cv2
        from detection.face_module_v7 import FaceAsymmetryDetector
        task = big[0] if big else (cands[0] if cands else None)
        det = FaceAsymmetryDetector(model_path=task)
        img = None
        for sub in ('NonStroke', 'Stroke'):
            for jp in sorted(glob.glob(os.path.join(FACE_DS, sub, '*.jpg')))[:5]:
                img = cv2.imread(jp)
                if img is not None:
                    break
            if img is not None:
                break
        check('đọc được ảnh mặt thật', img is not None)
        res = det.process_frame(cv2.resize(img, (640, 480)))
        st = res.get('status', '?')
        check('status ≠ NO_DETECTOR (không rơi numpy-mode)',
              st not in ('NO_DETECTOR', 'ERROR', 'INVALID_LANDMARKS'), st)
    except Exception as e:
        check('face detector chạy', False, str(e)[:80])

    # 3. Arm: khung trống KHÔNG crash + khung ảnh thật KHÔNG crash (bug 08/09)
    print('[3] Arm module (bug index 2 — 08/09):')
    try:
        import cv2
        from detection.arm_module import ArmWeaknessDetector
        arm = ArmWeaknessDetector(
            pose_model_path=os.path.join(SRC, 'yolov8n-pose.pt'),
            ml_model_path=os.path.join(
                MODELS, 'arm_weakness_20260830_200657.pth'),
            scaler_path=os.path.join(
                MODELS, 'arm_weakness_20260830_200657_scaler.pkl'))
        blank = np.zeros((480, 640, 3), np.uint8)
        r1 = arm.detect_arm_weakness(blank)
        check('khung trống → không crash', r1.get('status') == 'NO_PERSON',
              r1.get('status'))
        face_img = None
        for jp in sorted(glob.glob(os.path.join(FACE_DS, '*', '*.jpg')))[:3]:
            face_img = cv2.imread(jp)
            if face_img is not None:
                break
        if face_img is not None:
            r2 = arm.detect_arm_weakness(cv2.resize(face_img, (640, 480)))
            check('ảnh thật → không crash (bất kể status)',
                  r2.get('status') in ('NO_PERSON', 'NO_POSE', 'NORMAL',
                                       'WARNING', 'DANGER'),
                  r2.get('status'))
    except Exception as e:
        check('arm module chạy', False, str(e)[:80])

    # 4. Speech: 48 features có pitch (bug numpy 2.5 — 08/09)
    print('[4] Speech features (bug numba/numpy — 08/09):')
    try:
        from detection.speech_module_v2 import SpeechAnalysisModule
        m = SpeechAnalysisModule()
        t = np.arange(16000, dtype=np.float32) / 16000
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        feat, fd = m.extract_features(audio)
        check('có pitch_mean (librosa/numba sống)',
              isinstance(fd, dict) and 'pitch_mean' in fd)
        check('đủ 48 features', len(feat) == 48, f'{len(feat)} phần tử')
    except Exception as e:
        check('speech extract_features', False, str(e)[:80])

    # 5. Gait + radar instantiate
    print('[5] Gait + Radar:')
    try:
        from detection.gait_module import GaitPoseDetector
        g = GaitPoseDetector(
            model_path=os.path.join(SRC, 'yolov8n-pose.pt'))
        r = g.detect_gait_from_frame(np.zeros((480, 640, 3), np.uint8))
        check('gait YOLO load + frame trống', r.get('status') == 'NO_PERSON',
              r.get('status'))
    except Exception as e:
        check('gait module', False, str(e)[:80])
    try:
        from detection.radar_module import RadarModule
        rm = RadarModule()  # tự fallback SIM nếu chưa cắm
        r = rm.analyze(duration_s=0.2, sample_hz=3.0)
        check('radar analyze', r.get('status') in
              ('NORMAL', 'WARNING', 'DANGER'), f"sim={rm.simulation}")
    except Exception as e:
        check('radar module', False, str(e)[:80])

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\nTỔNG KẾT: {len(results) - n_fail}/{len(results)} PASS")
    if n_fail:
        print('⛔ KHÔNG DEMO — sửa trước khi nghiệm thu (S0 KICH_BAN_NGHIEM_THU)')
        sys.exit(1)
    print('✅ Sẵn sàng demo/nghiệm thu.')
    sys.exit(0)


if __name__ == '__main__':
    main()
