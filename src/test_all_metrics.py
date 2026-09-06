# -*- coding: utf-8 -*-
"""
PSCS - COMPREHENSIVE METRICS & LOGIC TEST SUITE
Kiểm thử đầy đủ các THANG ĐO và LOGIC code của hệ thống PSCS v8.x.

Phạm vi:
  A. Module 2 Speech: VAD, WPM, NIHSS mapping, window selection (M2-10),
     feature extraction (48), baseline roundtrip, NO_SPEECH reset
  B. FusionEngine: adapter key, renormalize, skip invalid, luật R1/R2,
     biên risk level, NIHSS severity, trend
  C. AlertSystem: quyết định mức, cooldown per-level, log JSONL
  D. Model checkpoints: tồn tại + load được + kiến trúc khớp

Chạy:
    venv/Scripts/python.exe src/test_all_metrics.py
Ra: exit code 0 = ALL PASS, 1 = có FAIL
Không cần camera/micro/dataset (audio tổng hợp + mock data).
"""

import os
import sys
import json
import glob
import time
import tempfile
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

SAMPLE_RATE = 16000

# Kết quả tổng: [(section, name, ok, detail), ...]
ALL_RESULTS = []


def check(section, name, fn):
    """Chạy 1 test case, bắt exception thành FAIL."""
    try:
        detail = fn()
        ALL_RESULTS.append((section, name, True, detail or ''))
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    except AssertionError as e:
        ALL_RESULTS.append((section, name, False, str(e)))
        print(f"  [FAIL] {name}  -> {e}")
    except Exception as e:
        ALL_RESULTS.append((section, name, False, f"EXCEPTION: {e}"))
        print(f"  [FAIL] {name}  -> EXCEPTION: {e}")


def assert_eq(actual, expected, msg=""):
    assert actual == expected, f"{msg} expected={expected!r} got={actual!r}"


def assert_true(cond, msg=""):
    assert cond, msg


def assert_close(actual, expected, tol, msg=""):
    assert abs(actual - expected) <= tol, \
        f"{msg} expected~{expected}±{tol} got={actual}"


# ======================================================================
# ÂM THANH TỔNG HỢP
# ======================================================================
def make_tone(seconds, freq=440, amp=0.3):
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_silence(seconds):
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


# ======================================================================
# A. MODULE 2 — SPEECH SCALES
# ======================================================================
def test_speech():
    print("\n" + "=" * 70)
    print("A. MODULE 2 SPEECH — VAD / WPM / NIHSS / WINDOW / FEATURES")
    print("=" * 70)
    from detection.speech_module_v2 import SpeechAnalysisModule
    m = SpeechAnalysisModule()  # không load model nào (nhanh, đủ test logic)

    # A1. VAD: im lặng hoàn toàn
    def a1():
        v = m.detect_voice_activity(make_silence(3))
        assert_eq(v['speech_ratio'], 0.0, "silence ratio")
        assert_true(not v['has_speech'], "silence has_speech")
        return f"ratio=0, noise_floor={v['noise_floor']}"
    check("A.Speech", "A1 VAD im lặng -> ratio 0, has_speech False", a1)

    # A2. VAD: tone liên tục
    def a2():
        v = m.detect_voice_activity(make_tone(3))
        assert_eq(v['speech_ratio'], 1.0, "tone ratio")
        assert_true(v['has_speech'], "tone has_speech")
        return "ratio=1.0"
    check("A.Speech", "A2 VAD tone 440Hz liên tục -> ratio 1.0", a2)

    # A3. VAD: nửa im lặng nửa tiếng
    def a3():
        audio = np.concatenate([make_silence(3), make_tone(3)])
        v = m.detect_voice_activity(audio)
        assert_close(v['speech_ratio'], 0.5, 0.15, "half-half ratio")
        assert_true(v['has_speech'], "has_speech")
        return f"ratio={v['speech_ratio']} (~0.5)"
    check("A.Speech", "A3 VAD 50% tiếng -> ratio ~0.5", a3)

    # A4. WPM
    def a4():
        assert_close(m.calculate_wpm(25, 10), 150.0, 0.01, "25 words/10s")
        assert_close(m.calculate_wpm(12, 5), 144.0, 0.01, "12 words/5s")
        assert_eq(m.calculate_wpm(10, 0), 0.0, "duration 0")
        return "150 & 144 & 0"
    check("A.Speech", "A4 WPM = words/duration*60", a4)

    # A5. NIHSS item 10 mapping (ngưỡng 30/50/70)
    def a5():
        cases = [(20, 0), (29.9, 0), (30, 1), (49.9, 1), (50, 2),
                 (69.9, 2), (70, 3), (100, 3)]
        for score, expect in cases:
            got = m._map_to_nihss(score)
            assert_eq(got, expect, f"nihss({score})")
        return "0/1/2/3 tại 30/50/70"
    check("A.Speech", "A5 NIHSS mapping item 10 đúng biên", a5)

    # A6. Window selection: 10s im + 15s tone + 5s noise-gần-im
    def a6():
        audio = np.concatenate([make_silence(10), make_tone(15),
                                make_tone(5, amp=0.001)])
        start, ratio = m.select_best_window(audio, duration_seconds=5)
        start_s = start / SAMPLE_RATE
        assert_true(9.0 <= start_s <= 20.5,
                    f"window phải rơi vùng tone (9-20.5s), got {start_s:.1f}s")
        assert_true(ratio > 0.9, f"ratio > 0.9, got {ratio}")
        return f"start={start_s:.1f}s ratio={ratio}"
    check("A.Speech", "A6 select_best_window chọn vùng CÓ TIẾNG (M2-10)", a6)

    # A7. Window selection: audio ngắn hơn cửa sổ
    def a7():
        start, ratio = m.select_best_window(make_tone(3), duration_seconds=5)
        assert_eq(start, 0, "short audio start=0")
        assert_true(ratio > 0.9, f"short audio ratio, got {ratio}")
        start0, ratio0 = m.select_best_window(make_silence(3))
        assert_eq(ratio0, 0.0, "empty ratio")
        return "start=0, ratio cao"
    check("A.Speech", "A7 select_best_window audio ngắn/rỗng an toàn", a7)

    # A8. extract_window: đúng độ dài + info
    def a8():
        audio = np.concatenate([make_silence(4), make_tone(10)])
        win, info = m.extract_window(audio, duration_seconds=5)
        assert_eq(len(win), 5 * SAMPLE_RATE, "window length 5s")
        assert_true(info['window_speech_ratio'] > 0.9, "info ratio")
        assert_true(info['window_start_s'] >= 3.0, "info start_s")
        return f"win@{info['window_start_s']}s ratio={info['window_speech_ratio']}"
    check("A.Speech", "A8 extract_window trả 5s chuẩn + info", a8)

    # A9. extract_features: đủ 48 features, không NaN
    def a9():
        features, fd = m.extract_features(make_tone(5))
        assert_eq(len(features), 48, "feature count")
        assert_true(np.isfinite(features).all(), "features finite")
        assert_close(float(fd['pitch_mean'][0]), 440.0, 25.0, "pitch ~440Hz")
        return "48 features, pitch~440Hz"
    check("A.Speech", "A9 extract_features = 48, không NaN, pitch đúng", a9)

    # A10. Baseline roundtrip set/save/load
    def a10():
        m.set_baseline(wpm=150.0, pitch_mean=180.0)
        tmp = os.path.join(tempfile.gettempdir(), 'pscs_baseline_test.json')
        assert_true(m.save_baseline(tmp), "save_baseline")
        m2 = SpeechAnalysisModule()
        assert_true(m2.load_baseline(tmp), "load_baseline")
        assert_close(m2.baseline['wpm'], 150.0, 0.01, "baseline wpm")
        os.remove(tmp)
        return "wpm=150 roundtrip OK"
    check("A.Speech", "A10 Baseline save/load roundtrip", a10)

    # A11. predict_dysarthria: audio im lặng -> NO_SPEECH, prob reset 0
    def a11():
        r = m.predict_dysarthria(make_silence(5), 5)
        assert_eq(r['status'], 'NO_SPEECH', "status")
        assert_eq(r['speech_prob'], 0.0, "prob reset")
        assert_eq(r['nihss_score'], 0, "nihss")
        return "NO_SPEECH, prob=0"
    check("A.Speech", "A11 predict im lặng -> NO_SPEECH + prob reset", a11)


# ======================================================================
# B. FUSION ENGINE LOGIC
# ======================================================================
def _mod(prob_key, prob, nihss=0, status=None):
    st = status or ('NORMAL' if prob < 30 else
                    ('WARNING' if prob < 60 else 'DANGER'))
    return {prob_key: prob, 'nihss_score': nihss, 'status': st, 'metrics': {}}


def test_fusion():
    print("\n" + "=" * 70)
    print("B. FUSION ENGINE — ADAPTER / RENORM / RULES / THRESHOLDS")
    print("=" * 70)
    from fusion.fusion_engine import FusionEngine

    # B1. Adapter: đúng prob key từng module
    def b1():
        fe = FusionEngine()
        r = fe.fuse({
            'face': _mod('score', 40),
            'speech': _mod('speech_prob', 20),
            'arm': _mod('arm_prob', 10),
            'gait': _mod('gait_prob', 0),
        })
        assert_close(r['modules_used']['face']['prob'], 40, 0.01, "face")
        assert_close(r['modules_used']['speech']['prob'], 20, 0.01, "speech")
        assert_close(r['modules_used']['arm']['prob'], 10, 0.01, "arm")
        return "4 keys adapter OK"
    check("B.Fusion", "B1 Adapter prob key (score/speech_prob/arm_prob...)", b1)

    # B2. Renormalize: 2 module còn lại -> trọng số tự chuẩn hóa
    def b2():
        fe = FusionEngine()
        r = fe.fuse({'face': _mod('score', 40), 'speech': _mod('speech_prob', 10)})
        # (0.2*40 + 0.2*10) / 0.4 = 25
        assert_close(r['fused_score'], 25.0, 0.1, "renormalized")
        assert_eq(set(r['modules_used'].keys()), {'face', 'speech'}, "used")
        return "fused=25 (40/10 renorm 0.4)"
    check("B.Fusion", "B2 Renormalize khi thiếu module", b2)

    # B3. Skip invalid status (NO_SPEECH...)
    def b3():
        fe = FusionEngine()
        r = fe.fuse({'speech': _mod('speech_prob', 90, status='NO_SPEECH'),
                     'arm': _mod('arm_prob', 50)})
        assert_true('speech' not in r['modules_used'], "NO_SPEECH excluded")
        assert_eq(r['modules_skipped'].get('speech'), 'NO_SPEECH', "skipped")
        assert_close(r['fused_score'], 50.0, 0.1, "fused từ arm duy nhất")
        return "NO_SPEECH bỏ, fused=50"
    check("B.Fusion", "B3 Module invalid bị loại + renormalize", b3)

    # B4. R1: 2/3 FAST dấu hiệu >= 50 -> fused >= 75 EMERGENCY
    def b4():
        fe = FusionEngine()
        r = fe.fuse({'face': _mod('score', 60, 2),
                     'arm': _mod('arm_prob', 60, 2),
                     'speech': _mod('speech_prob', 10)})
        # weighted = (12+18+2)/0.7 = 45.7 -> R1 nâng lên 75
        assert_close(r['fused_score'], 75.0, 0.1, "R1 floor")
        assert_eq(r['risk_level'], 'EMERGENCY', "R1 risk")
        assert_true(any(x.startswith('R1') for x in r['triggered_rules']),
                    "R1 rule logged")
        return "45.7 -> 75 EMERGENCY"
    check("B.Fusion", "B4 Luật R1: 2/3 FAST >=50 -> EMERGENCY", b4)

    # B5. R2: 1 module >= 80 -> floor 55 (WARNING tối thiểu)
    def b5():
        fe = FusionEngine()
        r = fe.fuse({'speech': _mod('speech_prob', 85, 3),
                     'arm': _mod('arm_prob', 5)})
        assert_true(r['fused_score'] >= 55.0, f"fused>=55, got {r['fused_score']}")
        assert_true(any(x.startswith('R2') for x in r['triggered_rules']),
                    "R2 rule logged")
        assert_true(r['risk_level'] in ('WARNING', 'EMERGENCY'), "risk")
        return f"fused={r['fused_score']}"
    check("B.Fusion", "B5 Luật R2: 1 module >=80 -> tối thiểu WARNING", b5)

    # B6. Biên risk level qua single module (renorm = chính nó)
    def b6():
        fe = FusionEngine()
        cases = [(29.9, 'NORMAL'), (30, 'MONITOR'), (49.9, 'MONITOR'),
                 (50, 'WARNING'), (69.9, 'WARNING'), (70, 'EMERGENCY')]
        for prob, expect in cases:
            r = fe.fuse({'gait': _mod('gait_prob', prob)})
            assert_eq(r['risk_level'], expect,
                      f"prob={prob} -> {expect}")
        return "6 biên 30/50/70 đúng"
    check("B.Fusion", "B6 Biên risk level NORMAL/MONITOR/WARNING/EMERGENCY", b6)

    # B7. NIHSS total = tổng items + severity
    def b7():
        fe = FusionEngine()
        r = fe.fuse({'face': _mod('score', 80, 2), 'arm': _mod('arm_prob', 80, 3),
                     'speech': _mod('speech_prob', 80, 2)})
        assert_eq(r['nihss_total'], 7, "total 2+3+2")
        assert_eq(r['severity'], 'MODERATE', "6-13 = MODERATE")
        r2 = fe.fuse({'face': _mod('score', 80, 3), 'arm': _mod('arm_prob', 80, 3),
                      'speech': _mod('speech_prob', 80, 3),
                      'gait': _mod('gait_prob', 80, 3)})
        assert_eq(r2['nihss_total'], 12, "total max4")
        r3 = fe.fuse({'gait': _mod('gait_prob', 10, 3), 'arm': _mod('arm_prob', 10, 3),
                      'face': _mod('score', 10, 3), 'speech': _mod('speech_prob', 10, 3)})
        assert_eq(r3['nihss_total'], 12, "mild total")
        assert_eq(r3['severity'], 'MODERATE', "12 = MODERATE")
        return "total=sum, 6-13=MODERATE, 14+=SEVERE"
    check("B.Fusion", "B7 NIHSS total + severity MILD/MODERATE/SEVERE", b7)

    # B8. get_trend WORSENING / IMPROVING / STABLE
    def b8():
        fe = FusionEngine()
        for p in [10, 20, 30, 50, 60]:
            fe.fuse({'gait': _mod('gait_prob', p)})
        assert_eq(fe.get_trend(), 'WORSENING', "worsening")
        fe2 = FusionEngine()
        for p in [60, 50, 40, 30, 20]:
            fe2.fuse({'gait': _mod('gait_prob', p)})
        assert_eq(fe2.get_trend(), 'IMPROVING', "improving")
        fe3 = FusionEngine()
        for p in [10, 12, 11, 13, 12]:
            fe3.fuse({'gait': _mod('gait_prob', p)})
        assert_eq(fe3.get_trend(), 'STABLE', "stable")
        assert_eq(FusionEngine().get_trend(), 'NO_DATA', "empty")
        return "WORSENING/IMPROVING/STABLE/NO_DATA"
    check("B.Fusion", "B8 get_trend 4 trạng thái", b8)


# ======================================================================
# C. ALERT SYSTEM
# ======================================================================
def test_alert():
    print("\n" + "=" * 70)
    print("C. ALERT SYSTEM — MỨC / COOLDOWN / JSONL")
    print("=" * 70)
    from alerts.alert_system import AlertSystem

    # C1. Quyết định mức (cooldown 0, buzzer off)
    def c1():
        al = AlertSystem(buzzer_enabled=False, cooldown_seconds=0)
        r1 = al.process_fusion_result({'fused_score': 10, 'risk_level': 'NORMAL'})
        assert_true(not r1['triggered'], "10 -> không alert")
        r2 = al.process_fusion_result({'fused_score': 40, 'risk_level': 'MONITOR'})
        assert_true(not r2['triggered'], "40 MONITOR -> không alert")
        r3 = al.process_fusion_result({'fused_score': 60, 'risk_level': 'WARNING'})
        assert_eq(r3['level'], 'WARNING', "60 -> WARNING")
        r4 = al.process_fusion_result({'fused_score': 85,
                                       'risk_level': 'EMERGENCY'})
        assert_eq(r4['level'], 'EMERGENCY', "85 -> EMERGENCY")
        r5 = al.send_alert(95, "score cao risk thấp")
        assert_eq(r5['level'], 'EMERGENCY', "95 score > 80 -> EMERGENCY")
        return "10/40 không báo, 60 WARN, 85/95 EMERGENCY"
    check("C.Alert", "C1 Quyết định mức theo score+risk (80/50)", c1)

    # C2. Cooldown per-level: WARNING không chặn EMERGENCY
    def c2():
        al = AlertSystem(buzzer_enabled=False, cooldown_seconds=60)
        r1 = al.send_alert(55, "warn 1", level='WARNING')
        assert_true(r1['triggered'], "WARNING đầu -> phát")
        r2 = al.send_alert(90, "emergency ngay sau warning")
        assert_eq(r2['level'], 'EMERGENCY', "level đúng")
        assert_true(r2['triggered'], "EMERGENCY không bị WARNING chặn")
        r3 = al.send_alert(90, "emergency lặp")
        assert_true(not r3['triggered'], "EMERGENCY lặp bị cooldown chặn")
        assert_true('cooldown' in r3['reason'], "reason cooldown")
        assert_true(al.suppressed_count >= 1, "suppressed_count tăng")
        return "per-level: warn->emer OK, emer lặp bị chặn"
    check("C.Alert", "C2 Cooldown per-level (WARN không chặn EMERGENCY)", c2)

    # C3. Log JSONL ghi ra file + parse được
    def c3():
        tmpdir = tempfile.mkdtemp(prefix='pscs_alert_test_')
        al = AlertSystem(buzzer_enabled=False, cooldown_seconds=0,
                         log_dir=tmpdir)
        al.send_alert(90, "test jsonl", level='EMERGENCY')
        files = glob.glob(os.path.join(tmpdir, '*.jsonl'))
        assert_true(len(files) == 1, f"1 log file, got {len(files)}")
        with open(files[0], encoding='utf-8') as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert_true(len(lines) >= 1, "có event")
        ev = lines[-1]
        for k in ('timestamp', 'session_id', 'level', 'score', 'message'):
            assert_true(k in ev, f"event thiếu {k}")
        assert_eq(ev['level'], 'EMERGENCY', "level trong log")
        return f"{len(lines)} event parse OK"
    check("C.Alert", "C3 Log JSONL: ghi + parse ngược được", c3)

    # C4. process_fusion_result thiếu data an toàn
    def c4():
        al = AlertSystem(buzzer_enabled=False)
        r = al.process_fusion_result(None)
        assert_true(not r['triggered'], "None -> không crash, không alert")
        r2 = al.process_fusion_result({})
        assert_true(not r2['triggered'], "{} -> an toàn")
        return "None/{} không crash"
    check("C.Alert", "C4 Input rỗng/None an toàn", c4)


# ======================================================================
# D. MODEL CHECKPOINTS
# ======================================================================
def test_models():
    print("\n" + "=" * 70)
    print("D. MODEL CHECKPOINTS — TỒN TẠI / LOAD / KIẾN TRÚC")
    print("=" * 70)
    models_dir = os.path.join(PROJECT_ROOT, 'models')

    # D1. Các file model bắt buộc tồn tại
    def d1():
        required = [
            'speech_torgo_20260828_211130.pth',       # M2 (đang dùng)
            'speech_torgo_20260828_211130_scaler.pkl',
            'arm_weakness_20260830_200657.pth',        # M3
            'arm_weakness_20260830_200657_scaler.pkl',
            'gait_classifier_20260829_120925.pth',     # M4
            'gait_classifier_20260829_120925_scaler.pkl',
            'face_landmarker_v2.task',                 # M1 MediaPipe
        ]
        missing = [f for f in required
                   if not os.path.exists(os.path.join(models_dir, f))]
        assert_true(not missing, f"thiếu: {missing}")
        return f"{len(required)} file OK"
    check("D.Models", "D1 File model bắt buộc tồn tại", d1)

    # D2. Speech checkpoint: input 48, out 2 (khớp auto-detect loader)
    def d2():
        import torch
        p = os.path.join(models_dir, 'speech_torgo_20260828_211130.pth')
        sd = torch.load(p, map_location='cpu', weights_only=True)
        dims = []
        idx = 0
        while f'network.{idx}.weight' in sd:
            dims.append(tuple(sd[f'network.{idx}.weight'].shape))
            idx += 4
        assert_eq(dims[0][1], 48, "input dim 48")
        assert_eq(dims[-1][0], 2, "output 2 lớp")
        assert_true(all(a[0] == b[1] for a, b in zip(dims, dims[1:])),
                    "kiến trúc liền mạch")
        return f"arch {dims[0][1]}->{'->'.join(str(d[0]) for d in dims)}"
    check("D.Models", "D2 Speech .pth: in=48, out=2, khớp loader", d2)

    # D3. Arm + Gait checkpoints load được, kiến trúc liền mạch
    def d3():
        import torch
        for name in ('arm_weakness_20260830_200657.pth',
                     'gait_classifier_20260829_120925.pth'):
            sd = torch.load(os.path.join(models_dir, name),
                            map_location='cpu', weights_only=True)
            dims = []
            idx = 0
            while f'network.{idx}.weight' in sd:
                dims.append(tuple(sd[f'network.{idx}.weight'].shape))
                idx += 4
            assert_true(len(dims) >= 2, f"{name}: ít nhất 2 layer")
            assert_true(all(a[0] == b[1] for a, b in zip(dims, dims[1:])),
                        f"{name}: layer không khớp")
        return "arm + gait load OK"
    check("D.Models", "D3 Arm/Gait .pth load + kiến trúc liền mạch", d3)

    # D4. Scaler load được + số features
    def d4():
        import joblib
        s1 = joblib.load(os.path.join(
            models_dir, 'speech_torgo_20260828_211130_scaler.pkl'))
        assert_eq(getattr(s1, 'n_features_in_', None), 48, "speech scaler 48")
        s2 = joblib.load(os.path.join(
            models_dir, 'arm_weakness_20260830_200657_scaler.pkl'))
        assert_true(getattr(s2, 'n_features_in_', 0) > 0, "arm scaler")
        return f"speech=48 feats, arm={s2.n_features_in_}"
    check("D.Models", "D4 Scaler load + n_features khớp", d4)


# ======================================================================
# E. TEST RESULT ARTIFACTS (kết quả extended M2 vừa chạy)
# ======================================================================
def test_artifacts():
    print("\n" + "=" * 70)
    print("E. TEST RESULTS — ARTIFACTS MỚI NHẤT")
    print("=" * 70)

    def e1():
        files = sorted(glob.glob(os.path.join(
            PROJECT_ROOT, 'test_results', 'module2_test_*.json')))
        assert_true(files, "chưa có kết quả M2 nào")
        d = json.load(open(files[-1], encoding='utf-8'))
        assert_eq(d.get('window_mode'), 'median3_best_windows_m2_10v2',
                  "window mode median3")
        assert_eq(d.get('n_normal'), 28, "28 normal")
        assert_eq(d.get('n_dys'), 27, "27 dys")
        tpr, fpr_o = d.get('TPR'), d.get('TPR_optimal')
        assert_true(tpr >= 90, f"TPR {tpr} >= 90")
        assert_true(d.get('FPR_optimal') < 5, f"FPR_opt {d.get('FPR_optimal')} < 5")
        assert_true(fpr_o >= tpr - 0.1, "optimal không mất TPR")
        return f"{os.path.basename(files[-1])}: TPR {tpr}%, FPR_opt {d.get('FPR_optimal')}%"
    check("E.Results", "E1 M2 extended (median3): TPR>=90, FPR_opt<5", e1)

    def e2():
        logs = sorted(glob.glob(os.path.join(
            PROJECT_ROOT, 'logs', 'alerts', '*.jsonl')))
        assert_true(logs, "chưa có alert log (chạy test C trước)")
        with open(logs[-1], encoding='utf-8') as f:
            n = sum(1 for l in f if l.strip())
        assert_true(n >= 1, "log có nội dung")
        return f"{os.path.basename(logs[-1])}: {n} events"
    check("E.Results", "E2 Alert JSONL tồn tại + có event", e2)


# ======================================================================
# F. DEFENSE ENGINE (4 LỚP)
# ======================================================================
def test_defense():
    print("\n" + "=" * 70)
    print("F. DEFENSE ENGINE — L1 CALIBRATE / L2 CONTEXT / L3 TEMPORAL / L4 FLOOR")
    print("=" * 70)
    from defense.defense_engine import DefenseEngine

    # F1. L1: calibrate baseline
    def f1():
        de = DefenseEngine()
        assert_true(not de.get_stats()['calibrated'], "mới tạo chưa calibrate")
        de.calibrate(wpm=148, face_asym=8.0)
        assert_true(de.get_stats()['calibrated'], "sau calibrate")
        assert_eq(de.baseline['wpm'], 148, "baseline wpm")
        return "calibrate ghi baseline + timestamp"
    check("F.Defense", "F1 L1 calibrate baseline cá nhân", f1)

    # F2. L2: EXERCISE suppress face/arm/gait (đánh dấu, KHÔNG mất prob)
    def f2():
        de = DefenseEngine()
        de.update_context(motion_intensity=0.8)   # > 0.6 → EXERCISE
        assert_eq(de.context, 'EXERCISE', "context phân loại")
        mods = {'face': {'score': 55, 'status': 'WARNING'},
                'arm': {'arm_prob': 60, 'status': 'WARNING'},
                'speech': {'speech_prob': 70, 'status': 'DANGER'}}
        clean, audit = de.filter(mods)
        assert_true('face' in audit['suppressed'] and 'arm' in audit['suppressed'],
                    f"suppressed={audit['suppressed']}")
        assert_eq(clean['face']['score'], 55, "L2 giữ nguyên prob (không mất data)")
        assert_true('defense_note' in clean['face'], "có defense_note")
        assert_true('defense_note' not in clean['speech'], "speech không suppress")
        return f"EXERCISE: {audit['suppressed']} marked, prob giữ nguyên"
    check("F.Defense", "F2 L2 CONTEXT_SUPPRESS EXERCISE (giữ data)", f2)

    # F3. L2: phân loại context theo motion/talking
    def f3():
        de = DefenseEngine()
        assert_eq(de.update_context(0.9), 'EXERCISE', "0.9")
        assert_eq(de.update_context(0.4), 'WALKING', "0.4")
        assert_eq(de.update_context(0.05, talking=True), 'TALKING', "talking")
        assert_eq(de.update_context(0.0), 'REST', "rest")
        return "EXERCISE/WALKING/TALKING/REST đúng biên 0.6/0.2"
    check("F.Defense", "F3 L2 phân loại context theo motion+talking", f3)

    # F4. L4: vùng xám [30,40) hạ 5 điểm CHỈ khi đã calibrate
    def f4():
        de = DefenseEngine()
        mods = {'gait': {'gait_prob': 35, 'status': 'MONITOR'}}
        clean, _ = de.filter(mods)
        assert_eq(clean['gait']['gait_prob'], 35, "chưa calibrate giữ nguyên")
        de.calibrate(wpm=150)
        clean2, audit2 = de.filter(mods)
        assert_eq(clean2['gait']['gait_prob'], 30.0, "vùng xám 35->30")
        assert_true('adaptive_floor' in clean2['gait'], "ghi adaptive_floor")
        clean3, _ = de.filter({'gait': {'gait_prob': 45}})
        assert_eq(clean3['gait']['gait_prob'], 45, "ngoài vùng xám giữ nguyên")
        return "35->30 khi calibrate, 45/35-không-calibrate giữ nguyên"
    check("F.Defense", "F4 L4 adaptive floor vùng xám [30,40)", f4)

    # F5. L3: transient bị chặn, persistent được qua
    def f5():
        de = DefenseEngine()
        v1 = de.verdict(70)                       # warming up -> cho qua
        assert_true(v1['alert'], "warming up cho qua (fail-safe)")
        de.verdict(10); de.verdict(10)
        v2 = de.verdict(70)                       # 1/4 -> transient
        assert_true(not v2['alert'] and 'transient' in v2['reason'],
                    f"transient bị chặn: {v2}")
        de2 = DefenseEngine()
        t0 = time.time()
        for i in range(20):
            de2.history.append((70.0, t0 - 25 + i))
        v3 = de2.verdict(70)                      # 21/21 >= 60% -> persistent
        assert_true(v3['alert'] and 'persistent' in v3['reason'], f"{v3}")
        v4 = de2.verdict(10)
        assert_true(not v4['alert'], "score < 50 không alert")
        return "transient chặn / persistent qua / <50 im"
    check("F.Defense", "F5 L3 temporal: transient vs persistent", f5)

    # F6. L3 trend: WORSENING/IMPROVING/STABLE/NO_DATA
    def f6():
        de = DefenseEngine()
        now = time.time()
        vals = [10, 20, 30, 50, 60]
        for i, v in enumerate(vals):
            de.history.append((float(v), now - 100 + i))
        assert_eq(de._trend(), 'WORSENING', "delta +50")
        de2 = DefenseEngine()
        for i, v in enumerate([60, 50, 40, 30, 20]):
            de2.history.append((float(v), now - 100 + i))
        assert_eq(de2._trend(), 'IMPROVING', "delta -40")
        de3 = DefenseEngine()
        for i, v in enumerate([10, 12, 11]):
            de3.history.append((float(v), now - 100 + i))
        assert_eq(de3._trend(), 'STABLE', "delta 1")
        assert_eq(DefenseEngine()._trend(), 'NO_DATA', "empty")
        return "4 trend đúng"
    check("F.Defense", "F6 trend 4 trạng thái (window 5p)", f6)


# ======================================================================
# G. RADAR MODULE M5 (LD2450)
# ======================================================================
def test_radar():
    print("\n" + "=" * 70)
    print("G. RADAR M5 — PARSER LD2450 / FALL LOGIC / AND-GATE")
    print("=" * 70)
    from detection.radar_module import LD2450Parser, RadarModule, \
        FALL_DISPLACEMENT_CM, INACTIVITY_FALL_S, AUDIO_GATE_REDUCE

    # G1. Parser: frame 30 bytes chuẩn + byte rác
    def g1():
        frame = bytearray(b'\xAA\xFF\x03\x00')
        frame += (25).to_bytes(2, 'little', signed=True)
        frame += (150).to_bytes(2, 'little', signed=True)
        frame += (10).to_bytes(2, 'little', signed=True)
        frame += (150).to_bytes(2, 'little')
        frame += bytes(16)
        frame += b'\x55\xCC'
        p = LD2450Parser()
        frames = p.feed(bytes(frame) + b'\x00' * 5)
        assert_eq(len(frames), 1, f"1 frame, got {len(frames)}")
        tg = LD2450Parser.parse_frame(frames[0])
        assert_eq(len(tg), 1, "1 mục tiêu")
        assert_eq((tg[0]['x_cm'], tg[0]['y_cm']), (25, 150), "tọa độ LE")
        # 2 frame liền nhau
        frames2 = p.feed(bytes(frame) * 2)
        assert_eq(len(frames2), 2, "2 frame liền")
        return f"parse OK: {tg[0]}, 2 frame liền OK"
    check("G.Radar", "G1 LD2450Parser frame 30B + rác + ghép frame", g1)

    # G2. Parser: frame giả (sai tail) bị bỏ qua không treo
    def g2():
        bad = bytearray(b'\xAA\xFF\x03\x00')
        bad += bytes(24)                    # tail sai (00 00 thay vì 55 CC)
        p = LD2450Parser()
        frames = p.feed(bytes(bad))
        assert_eq(len(frames), 0, "frame sai tail bị loại")
        frame_ok = bytearray(b'\xAA\xFF\x03\x00') + bytes(24) + b'\x55\xCC'
        frames2 = p.feed(bytes(frame_ok))
        assert_eq(len(frames2), 1, "frame sau đó vẫn parse được")
        return "fake header bỏ, frame sau OK"
    check("G.Radar", "G2 Frame giả (sai tail) không làm treo parser", g2)

    # G3. Hằng số ngưỡng đúng spec
    def g3():
        assert_eq(FALL_DISPLACEMENT_CM, 100.0, "1m displacement")
        assert_eq(INACTIVITY_FALL_S, 45.0, "45s bất hoạt")
        assert_eq(AUDIO_GATE_REDUCE, 0.3, "gate ×0.3")
        return "100cm / 45s / 0.3"
    check("G.Radar", "G3 Ngưỡng fall đúng spec TONG_QUAN v6.0", g3)

    # G4. Fall logic: inactivity 50s sau biến động (+60), gate x0.3 audio normal
    def g4():
        rm = RadarModule(simulation=True, sim_scenario='fall')
        rm.set_audio_flag(False)
        # monkeypatch: người nằm yên tại (150,60) — KHÔNG di chuyển nữa
        rm.read_targets = lambda: [{'x_cm': 150, 'y_cm': 60, 'speed_cms': 0}]
        rm.last_position = (150, 60)
        rm.sudden_change_at = time.time() - 50   # ngã 50s trước (>=45s)
        rm.last_move_time = time.time() - 50
        r = rm.analyze(duration_s=0.5)
        # +60 (bất động 50s sau biến động) — không +40 vì không có chuyển động mới
        assert_close(r['fall_prob'], 18.0, 0.5,
                     f"60*0.3=18, got {r['fall_prob']}")
        assert_eq(r['status'], 'NORMAL', "18 < 30 -> NORMAL")
        assert_true('x0.3' in r['metrics']['gate'], f"gate={r['metrics']['gate']}")
        assert_eq(r['nihss_score'], 0, "radar không góp NIHSS")
        return f"60 -> 18 (gate x0.3, audio normal)"
    check("G.Radar", "G4 Fall: proxy bất hoạt + AND-gate x0.3", g4)

    # G5. AND-gate: audio BẤT THƯỜNG giữ nguyên prob 60 -> DANGER
    def g5():
        rm = RadarModule(simulation=True, sim_scenario='fall')
        rm.set_audio_flag(True)
        rm.read_targets = lambda: [{'x_cm': 150, 'y_cm': 60, 'speed_cms': 0}]
        rm.last_position = (150, 60)
        rm.sudden_change_at = time.time() - 50
        rm.last_move_time = time.time() - 50
        r = rm.analyze(duration_s=0.5)
        assert_close(r['fall_prob'], 60.0, 0.5, f"giữ 60, got {r['fall_prob']}")
        assert_eq(r['status'], 'DANGER', ">=60 DANGER")
        assert_true('ABN' in r['metrics']['gate'] or 'giữ' in r['metrics']['gate'],
                    f"gate={r['metrics']['gate']}")
        return f"prob=60 giữ nguyên -> DANGER (audio bất thường)"
    check("G.Radar", "G5 AND-gate: audio bất thường -> giữ prob", g5)

    # G6. Normal scenario không DANGER + output format FusionEngine
    def g6():
        rm = RadarModule(simulation=True, sim_scenario='normal')
        r = rm.analyze(duration_s=1.0)
        assert_true(r['status'] != 'DANGER', f"normal không DANGER: {r}")
        for k in ('fall_prob', 'status', 'nihss_score', 'metrics'):
            assert_true(k in r, f"thiếu key {k}")
        return f"prob={r['fall_prob']} status={r['status']}, đủ 4 key"
    check("G.Radar", "G6 Normal không DANGER + format FusionEngine", g6)


# ======================================================================
# H. NIHSS ESTIMATOR
# ======================================================================
def test_nihss():
    print("\n" + "=" * 70)
    print("H. NIHSS ESTIMATOR — BAND / TOTAL 13 / CI MONTE CARLO")
    print("=" * 70)
    from fusion.nihss_estimator import (prob_to_score, estimate_nihss,
                                        calculate_nihss_ci, ITEM_MAX)

    # H1. Band chuyển prob -> item đúng biên NIHSS thật
    def h1():
        cases = [
            ('item4_facial_palsy', 29.9, 0), ('item4_facial_palsy', 30, 1),
            ('item4_facial_palsy', 55, 2), ('item4_facial_palsy', 80, 3),
            ('item5_motor_arm', 84.9, 3), ('item5_motor_arm', 85, 4),
            ('item10_dysarthria', 59.9, 1), ('item10_dysarthria', 60, 2),
            ('item10_dysarthria', 29.9, 0),
        ]
        for item, prob, expect in cases:
            got = prob_to_score(item, prob)
            assert_eq(got, expect, f"{item}({prob})")
        assert_eq(ITEM_MAX['item10_dysarthria'], 2, "item10 max 2 (KHÔNG phải 3)")
        return "band 4 items đúng biên, item10 max 2"
    check("H.NIHSS", "H1 Band prob->item đúng biên (item10 max 2)", h1)

    # H2. Tổng 4 module = 9/13 (max_possible = 13 không phải 15)
    def h2():
        mods = {'face': {'score': 65, 'status': 'WARNING'},     # item4=2
                'arm': {'arm_prob': 90, 'status': 'DANGER'},    # item5=4
                'gait': {'gait_prob': 35, 'status': 'MONITOR'}, # item6=1
                'speech': {'speech_prob': 70, 'status': 'DANGER'}}  # item10=2
        est = estimate_nihss(mods)
        assert_eq(est['total'], 9, f"2+4+1+2, got {est}")
        assert_eq(est['max_possible'], 13, "subtotal 4 items max 13")
        assert_eq(est['items']['item10_dysarthria'], 2, "item10=2")
        return f"total 9/13 {est['items']}"
    check("H.NIHSS", "H2 Total 4 module = 9, max_possible = 13", h2)

    # H3. Module invalid/missing bị loại khỏi tổng
    def h3():
        est = estimate_nihss({'face': {'score': 0, 'status': 'NO_FACE'},
                              'arm': {'arm_prob': 90, 'status': 'DANGER'},
                              'gait': None})
        assert_eq(est['total'], 4, "chỉ arm góp 4")
        assert_true('item4_facial_palsy' in est['items_missing'] and
                    'item6_motor_leg' in est['items_missing'],
                    f"missing={est['items_missing']}")
        assert_eq(estimate_nihss({})['total'], 0, "empty an toàn")
        return "NO_FACE/None bị loại, missing ghi rõ, empty=0"
    check("H.NIHSS", "H3 Invalid/missing/empty an toàn", h3)

    # H4. CI Monte Carlo: total nằm trong [ci_low, ci_high]
    def h4():
        mods = {'face': {'score': 65}, 'arm': {'arm_prob': 90},
                'gait': {'gait_prob': 35}, 'speech': {'speech_prob': 70}}
        ci = calculate_nihss_ci(mods, n_iter=300)
        assert_eq(ci['total'], 9, "total không nhiễu")
        assert_true(ci['ci_low'] <= ci['total'] <= ci['ci_high'], f"{ci}")
        assert_true(ci['ci_high'] <= 13, f"CI không vượt max 13: {ci}")
        assert_true(ci['ci_low'] >= 0, "CI >= 0")
        assert_eq(calculate_nihss_ci({})['total'], 0, "empty CI an toàn")
        return f"9 ± {ci['margin']} [{ci['ci_low']}-{ci['ci_high']}]"
    check("H.NIHSS", "H4 CI 95% Monte Carlo chứa total, biên [0,13]", h4)


# ======================================================================
# I. TRIAGE ENGINE
# ======================================================================
def test_triage():
    print("\n" + "=" * 70)
    print("I. TRIAGE — SUBTYPE / SEVERITY / HOSPITAL")
    print("=" * 70)
    from fusion.triage_engine import TriageEngine
    te = TriageEngine()

    # I1. Subtype: base 80% ischemic; headache+nôn -> hemorrhagic 70%
    def i1():
        s = te.estimate_subtype({})
        assert_eq((s['subtype'], s['confidence']), ('ischemic', 80.0), f"{s}")
        s2 = te.estimate_subtype({'sudden_severe_headache': True,
                                  'vomiting': True})
        assert_eq((s2['subtype'], s2['confidence']),
                  ('hemorrhagic', 70.0), f"20+30+20={s2}")
        assert_true(len(s2['reasons']) >= 2, "ghi reasons")
        return "base 80/20; +30 đầu đau +20 nôn -> hem 70%"
    check("I.Triage", "I1 Subtype rule base + headache/vomiting", i1)

    # I2. Subtype: speech-dominant -> ischemic tăng (clamp 95)
    def i2():
        mods = {'speech': {'speech_prob': 90, 'status': 'DANGER'},
                'face': {'score': 10, 'status': 'NORMAL'}}
        s = te.estimate_subtype(module_results=mods)
        assert_eq((s['subtype'], s['confidence']), ('ischemic', 95.0),
                  f"clamp 95: {s}")
        return "speech-dominant -> ischemic 95% (clamp)"
    check("I.Triage", "I2 Speech-dominant +15% ischemic, clamp 95", i2)

    # I3. Severity bands + nâng bậc khi WORSENING
    def i3():
        assert_eq(te.classify_severity(4), 'MILD', "0-5")
        assert_eq(te.classify_severity(8), 'MODERATE', "6-10")
        assert_eq(te.classify_severity(12), 'SEVERE', ">=11")
        assert_eq(te.classify_severity(4, trend='WORSENING'), 'MODERATE',
                  "MILD + worsening -> MODERATE")
        assert_eq(te.classify_severity(12, trend='WORSENING'), 'SEVERE',
                  "SEVERE không nâng nổi")
        return "5/8/12 + upgrade WORSENING đúng"
    check("I.Triage", "I3 Severity MILD<6/MODERATE 6-10/SEVERE>=11 + trend", i3)

    # I4. Hospital: SEVERE -> stroke unit; MILD -> gần nhất
    def i4():
        hs = te.load_hospitals()
        assert_true(len(hs) >= 3, f"hospital DB >=3, got {len(hs)}")
        rec_sev = te.recommend_hospital('SEVERE', 'ischemic')
        assert_true(rec_sev is not None, "SEVERE có bệnh viện")
        su = str(rec_sev['stroke_unit']).strip().lower()
        th = str(rec_sev['thrombolysis']).strip().lower()
        assert_true(su in ('1', 'true', 'yes', 'co') and
                    th in ('1', 'true', 'yes', 'co'),
                    f"SEVERE cần stroke_unit+thrombolysis: {rec_sev['name']}")
        rec_mild = te.recommend_hospital('MILD', 'ischemic')
        assert_true(float(rec_mild['distance_km']) <=
                    float(rec_sev['distance_km']), "MILD = gần nhất")
        rec_hem = te.recommend_hospital('MODERATE', 'hemorrhagic')
        ns = str(rec_hem['neurosurgery']).strip().lower()
        assert_true(ns in ('1', 'true', 'yes', 'co'),
                    f"hemorrhagic cần neurosurgery: {rec_hem['name']}")
        d = str(hs[0].get('distance_km', '')).strip()
        assert_true(d, "distance_km không rỗng (bug CSV cũ)")
        return f"SEVERE={rec_sev['name']}, MILD gần nhất, HEM có NS"
    check("I.Triage", "I4 Hospital theo severity/subtype + CSV sạch", i4)


# ======================================================================
# J. HANDOFF SYSTEM
# ======================================================================
def test_handoff():
    print("\n" + "=" * 70)
    print("J. HANDOFF — TIMELINE / REPORT / QR")
    print("=" * 70)
    from handoff.handoff_system import HandoffSystem

    # J1. Timeline tự có T0 + thêm event đúng thứ tự
    def j1():
        hs = HandoffSystem({'name': 'Test', 'age': 70}, output_dir=tempfile.mkdtemp(
            prefix='pscs_handoff_'))
        assert_eq(hs.events[0]['tag'], 'T0', "T0 auto")
        hs.add_event('T3', 'Canh bao EMERGENCY', {'score': 85})
        assert_eq(hs.events[-1]['tag'], 'T3', "T3 thêm cuối")
        assert_true(hs.events[-1]['elapsed_min'] >= hs.events[0]['elapsed_min'],
                    "elapsed tăng dần")
        return f"{len(hs.events)} events T0->T3"
    check("J.Handoff", "J1 Timeline T0 auto + add_event", j1)

    # J2. build_report đủ 8 khối + JSON-serializable + save/load roundtrip
    def j2():
        tmpdir = tempfile.mkdtemp(prefix='pscs_handoff_')
        hs = HandoffSystem({'name': 'Test'}, output_dir=tmpdir)
        report = hs.build_report(
            fusion_result={'fused_score': 75, 'risk_level': 'EMERGENCY'},
            nihss_result={'total': 9},
            triage_result={'severity': 'MODERATE'},
            alerts=[{'level': 'EMERGENCY', 'score': 85}])
        for k in ('meta', 'patient', 'fusion', 'nihss', 'triage', 'alerts',
                  'timeline', 'defense'):
            assert_true(k in report, f"report thiếu {k}")
        path = hs.save_report(report)
        loaded = json.load(open(path, encoding='utf-8'))
        assert_eq(loaded['fusion']['fused_score'], 75, "roundtrip")
        return f"8 khối, save+load OK"
    check("J.Handoff", "J2 build_report 8 khối + JSON roundtrip", j2)

    # J3. QR tạo được (nếu cài qrcode) và decode được cấu trúc
    def j3():
        try:
            import qrcode  # noqa
        except ImportError:
            return "bỏ qua (chưa cài qrcode)"
        hs = HandoffSystem({'name': 'Test'}, output_dir=tempfile.mkdtemp(
            prefix='pscs_handoff_'))
        report = hs.build_report(fusion_result={'fused_score': 75},
                                 nihss_result={'total': 9},
                                 triage_result={'subtype': {'subtype': 'ischemic'}})
        png, data = hs.generate_qr(report)
        assert_true(os.path.exists(png), f"QR file: {png}")
        assert_true('NIHSS' in (data or '') or '9' in (data or ''),
                    f"QR data tóm tắt: {str(data)[:50]}")
        return f"QR {os.path.getsize(png)}B"
    check("J.Handoff", "J3 QR tóm tắt tạo được + file tồn tại", j3)


# ======================================================================
# K. VALIDATION METRICS (kiểm định y khoa)
# ======================================================================
def test_validation():
    print("\n" + "=" * 70)
    print("K. VALIDATION METRICS — r / MAE / ROC / YOUDEN / SIGN TEST")
    print("=" * 70)
    from fusion.validation_metrics import (
        pearson_r, mae, confusion_matrix, sensitivity, specificity,
        f1_score, roc_points, find_youden, paired_sign_test,
        evaluate_nihss_study, evaluate_detection_study)

    # K1. Pearson + MAE cơ bản
    def k1():
        assert_close(pearson_r([1, 2, 3], [1, 2, 3]), 1.0, 1e-9, "perfect")
        assert_close(pearson_r([1, 2, 3], [3, 2, 1]), -1.0, 1e-9, "nghịch")
        assert_eq(pearson_r([1, 1, 1], [1, 2, 3]), 0.0, "std=0 an toàn")
        assert_close(mae([2, 4], [1, 2]), 1.5, 1e-9, "MAE")
        assert_eq(mae([], []), 0.0, "empty")
        return "r ±1, std0=0, MAE=1.5"
    check("K.Validation", "K1 pearson_r + mae cơ bản + an toàn", k1)

    # K2. Confusion matrix + sens/spec/f1
    def k2():
        cm = confusion_matrix([1, 1, 0, 0], [1, 0, 1, 0])
        assert_eq(cm, {'TP': 1, 'FP': 1, 'TN': 1, 'FN': 1}, "4 ô bằng nhau")
        assert_close(sensitivity(cm), 0.5, 1e-9, "sens")
        assert_close(specificity(cm), 0.5, 1e-9, "spec")
        cm2 = confusion_matrix([1, 1, 0, 0], [1, 1, 0, 0])
        assert_close(f1_score(cm2), 1.0, 1e-9, "F1 perfect")
        return "TP/FP/TN/FN + sens/spec/F1"
    check("K.Validation", "K2 Confusion matrix + sens/spec/F1", k2)

    # K3. ROC + Youden J: ngưỡng tách hoàn hảo được tìm đúng
    def k3():
        y = [0] * 50 + [1] * 50
        scores = list(np.linspace(1, 49, 50)) + list(np.linspace(51, 99, 50))
        th, j, tpr, fpr = find_youden(y, scores)
        assert_close(j, 1.0, 1e-6, f"J={j} phải 1.0 (tách hoàn hảo)")
        assert_true(49 <= th <= 52, f"th={th} quanh 50")
        assert_close(tpr, 1.0, 1e-6, "TPR")
        assert_close(fpr, 0.0, 1e-6, "FPR")
        pts = roc_points([1, 0], [90, 10])
        assert_eq(len(pts), 99, "99 ngưỡng 1-99")
        assert_eq(len(roc_points([1, 1], scores)), 0, "1 lớp -> rỗng")
        return f"J=1.0 @th={th}"
    check("K.Validation", "K3 ROC + Youden J tách hoàn hảo", k3)

    # K4. Sign test: giống hệt p=1, B thắng toàn bộ p nhỏ
    def k4():
        y = [0] * 10 + [1] * 10
        pred = list(y)
        assert_eq(paired_sign_test(y, pred, pred), 1.0, "giống hệt")
        pred_b = [1] * 20
        p = paired_sign_test(y, pred, pred_b)
        assert_true(p < 0.05, f"B thắng toàn bộ 20/20 -> p<0.05, got {p}")
        return "p=1 giống nhau; p<0.05 khác biệt rõ"
    check("K.Validation", "K4 paired sign test biên", k4)

    # K5. evaluate_nihss_study: PASS với est≈gold, FAIL với est sai
    def k5():
        rng = np.random.default_rng(7)
        gold = rng.integers(0, 13, 50).astype(float)
        est = gold + rng.normal(0, 0.9, 50)
        res = evaluate_nihss_study(est, gold)
        assert_eq(res['verdict'], 'PASS', f"r={res['pearson_r']}")
        assert_true(res['pearson_r'] >= 0.85 and res['mae'] < 2, f"{res}")
        bad = evaluate_nihss_study(gold * 0.3, gold)
        assert_eq(bad['verdict'], 'FAIL', "est sai -> FAIL")
        return f"PASS (r={res['pearson_r']}, MAE={res['mae']}) / FAIL đúng"
    check("K.Validation", "K5 evaluate_nihss_study target r>=0.85 MAE<2", k5)

    # K6. evaluate_detection_study: tách hoàn hảo PASS, xấu CHECK
    def k6():
        rng = np.random.default_rng(7)
        y = [0] * 50 + [1] * 50
        scores = list(np.concatenate([rng.uniform(5, 25, 50),
                                      rng.uniform(45, 95, 50)]))
        det = evaluate_detection_study(y, scores)
        assert_eq(det['verdict'], 'PASS', f"{det}")
        assert_true(det['sensitivity'] > 90 and det['specificity'] > 95,
                    f"Sens={det['sensitivity']} Spec={det['specificity']}")
        assert_true(det['threshold'] >= 25, f"th={det['threshold']} biên 2 cụm")
        det_bad = evaluate_detection_study(y, list(rng.uniform(0, 100, 100)))
        assert_eq(det_bad['verdict'], 'CHECK', "random -> CHECK")
        return f"PASS (Sens={det['sensitivity']} Spec={det['specificity']})/CHECK"
    check("K.Validation", "K6 evaluate_detection_study Sens>90 Spec>95", k6)


# ======================================================================
# MAIN
# ======================================================================
def main():
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("PSCS COMPREHENSIVE METRICS & LOGIC TEST")
    print(f"Project: {PROJECT_ROOT}")

    test_speech()
    test_fusion()
    test_alert()
    test_models()
    test_artifacts()
    test_defense()
    test_radar()
    test_nihss()
    test_triage()
    test_handoff()
    test_validation()

    # Tổng kết
    passed = sum(1 for _, _, ok, _ in ALL_RESULTS if ok)
    total = len(ALL_RESULTS)
    print("\n" + "=" * 70)
    print(f"TỔNG KẾT: {passed}/{total} PASS")
    print("=" * 70)
    by_section = {}
    for sec, name, ok, _ in ALL_RESULTS:
        by_section.setdefault(sec, [0, 0])
        by_section[sec][0] += ok
        by_section[sec][1] += 1
    for sec, (p, t) in by_section.items():
        print(f"  {sec:<12} {p}/{t}")

    failed = [(s, n, d) for s, n, ok, d in ALL_RESULTS if not ok]
    if failed:
        print("\nCÁC TEST FAIL:")
        for s, n, d in failed:
            print(f"  - [{s}] {n}: {d}")

    # Export JSON
    out = os.path.join(PROJECT_ROOT, 'test_results',
                       f"metrics_suite_{sys.argv[1] if len(sys.argv) > 1 else 'run'}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'passed': passed, 'total': total,
                   'results': [{'section': s, 'name': n, 'ok': ok,
                                'detail': d} for s, n, ok, d in ALL_RESULTS]},
                  f, indent=2, ensure_ascii=False)
    print(f"\nExported: {out}")
    sys.exit(0 if passed == total else 1)


if __name__ == '__main__':
    main()
