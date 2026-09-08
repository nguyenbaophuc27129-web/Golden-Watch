# -*- coding: utf-8 -*-
"""
MODULE 2 FULL TEST - 10 NORMAL + 5 DYSARTHRIA (PSCS v8.0)
Test trên TORGO dataset THẬT (audio tiếng Anh, dysarthria là ngôn ngữ độc lập)

Target theo lịch ngày 04/09:
  - 10 normal speech (F_Con, M_Con)
  - 5 dysarthria speech (F_Dys, M_Dys)
  - FPR < 5%  (normal bị báo động)
  - TPR > 90% (dysarthria được phát hiện)

M2-10 (06/09): mặc định dùng VAD-guided selection 2 tầng:
  Tầng 1: chọn file WAV có speech_ratio cao nhất trong session
          (file _0001 headMic thường là calibration im lặng)
  Tầng 2: trong file đó, chọn cửa sổ 5s có speech_ratio cao nhất
  (giống phân phối train: câu đọc CÓ TIẾNG).
Cờ --first5 quay về logic cũ (5s đầu file đầu) để A/B so sánh.

Usage:
    venv/Scripts/python.exe module2/test_module2_full.py [--extended] [--first5]
"""

import os
import sys
import json
import glob
import numpy as np
import librosa
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_dir)

from detection.speech_module_v2 import SpeechAnalysisModule

# Config
SAMPLE_RATE = 16000
DURATION = 5
TORGO = os.path.join(parent_dir, "data/datasets/speech",
                     "TORGO Dataset for Dysarthric Speech - Audio Files")
VOSK_MODEL = os.path.join(parent_dir, "models/vosk-model-vn-0.4")
ML_MODEL = os.path.join(parent_dir, "models/speech_torgo_20260828_211130.pth")
SCALER = os.path.join(parent_dir, "models/speech_torgo_20260828_211130_scaler.pkl")
RESULTS_DIR = os.path.join(parent_dir, "test_results")

# 10 normal - chọn từ các session khác nhau (4 nữ + 6 nam)
NORMAL_SESSIONS = [
    "F_Con/wav_arrayMic_FC03S03",
    "F_Con/wav_headMic_FC03S01",
    "F_Con/wav_arrayMic_FC02S03",
    "F_Con/wav_headMic_FC03S03",
    "M_Con/wav_arrayMic_MC04S02",
    "M_Con/wav_headMic_MC04S01",
    "M_Con/wav_arrayMic_MC03S02",
    "M_Con/wav_headMic_MC03S02",
    "M_Con/wav_headMic_MC03S01",
    "M_Con/wav_arrayMic_MC04S01",
]

# 5 dysarthria - 2 nữ + 3 nam (speaker khác nhau)
DYS_SESSIONS = [
    "F_Dys/wav_arrayMic_F04S02",
    "F_Dys/wav_headMic_F03S03",
    "M_Dys/wav_arrayMic_M05S01",
    "M_Dys/wav_headMic_M04S02",
    "M_Dys/wav_arrayMic_M04S02",
]


def get_all_sessions():
    """Extended mode: TẤT CẢ các session (28 normal + 26 dys, 1 file/session)"""
    normal, dys = [], []
    for cat, out in [('F_Con', normal), ('M_Con', normal),
                     ('F_Dys', dys), ('M_Dys', dys)]:
        cat_dir = os.path.join(TORGO, cat)
        if os.path.isdir(cat_dir):
            for sess in sorted(os.listdir(cat_dir)):
                if os.path.isdir(os.path.join(cat_dir, sess)):
                    out.append(f"{cat}/{sess}")
    return normal, dys


def load_audio_5s(path):
    """Legacy (M2-10 --first5): load 5s đầu file — logic cũ trước khi fix"""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
    target = int(SAMPLE_RATE * DURATION)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    return audio[:target]


def load_audio_full(path):
    """M2-10: load FULL audio (16kHz) để chọn cửa sổ có tiếng nhất"""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    return audio


def first_wav(session):
    files = sorted(glob.glob(os.path.join(TORGO, session, "*.wav")))
    return files[0] if files else None


def best_session_wav(speech, session, max_files=10):
    """M2-10 tầng 1: chọn file WAV có speech_ratio cao nhất trong session.
    File _0001 của session headMic thường là calibration im lặng
    (ratio < 0.1) — không đại diện cho giọng của speaker."""
    files = sorted(glob.glob(os.path.join(TORGO, session, "*.wav")))[:max_files]
    best_path, best_ratio = None, -1.0
    for f in files:
        audio, _ = librosa.load(f, sr=SAMPLE_RATE)
        ratio = speech.detect_voice_activity(audio)['speech_ratio']
        if ratio > best_ratio:
            best_ratio, best_path = ratio, f
    return best_path, best_ratio


def run_session(speech, session, first5, n_windows=3):
    """Chạy 1 session.

    first5:  legacy — 5s đầu file đầu.
    else:    M2-10 v2 (multi-window consensus):
      Tầng 1: quét tối đa 5 file của session theo VAD ratio, giữ top 3
              cửa sổ (file, start) có tiếng nhất.
      Tầng 2: predict TỪNG cửa sổ, prob cuối = MEDIAN 3 giá trị.
      Median chống nhiễu đơn cửa sổ (1 cửa sổ bất thường không báo động).
    """
    if first5:
        path = first_wav(session)
        if not path:
            return None
        audio = load_audio_5s(path)
        r = speech.predict_dysarthria(audio, DURATION)
        return {'session': session, **{
            k: r[k] for k in ('status', 'speech_prob', 'nihss_score')},
            'wpm': round(r['metrics']['wpm'], 1),
            'speech_ratio': r['metrics'].get('speech_ratio'),
            'window_start_s': 0.0, 'window_ratio': None,
            'source_ratio': None, 'file': os.path.basename(path),
            'window_probs': None}

    # Tầng 1: thu ứng viên (file tốt nhất của từng lần quét top-5 files)
    files = sorted(glob.glob(os.path.join(TORGO, session, "*.wav")))[:5]
    candidates = []
    for f in files:
        audio_full = librosa.load(f, sr=SAMPLE_RATE)[0]
        src_ratio = speech.detect_voice_activity(audio_full)['speech_ratio']
        start, win_ratio = speech.select_best_window(audio_full, DURATION)
        candidates.append({'file': f, 'src_ratio': src_ratio,
                           'win_ratio': win_ratio, 'start': start})
    candidates.sort(key=lambda c: c['win_ratio'], reverse=True)
    top = candidates[:n_windows]

    # Tầng 2: predict từng cửa sổ -> median
    probs, rows = [], []
    for c in top:
        audio_full = librosa.load(c['file'], sr=SAMPLE_RATE)[0]
        win_len = int(SAMPLE_RATE * DURATION)
        audio = audio_full[c['start']:c['start'] + win_len]
        if len(audio) < win_len:
            audio = np.pad(audio, (0, win_len - len(audio)))
        r = speech.predict_dysarthria(audio, DURATION)
        probs.append(r['speech_prob'])
        rows.append({'file': os.path.basename(c['file']),
                     'start_s': round(c['start'] / SAMPLE_RATE, 1),
                     'win_ratio': c['win_ratio'],
                     'src_ratio': c['src_ratio'],
                     'prob': round(r['speech_prob'], 1)})
    probs_sorted = sorted(probs)
    median_prob = probs_sorted[len(probs_sorted) // 2] if probs else 0.0
    best = top[0]

    status = ('NORMAL' if median_prob < 30 else
              'WARNING' if median_prob < 60 else 'DANGER')
    nihss = 0 if median_prob < 30 else (1 if median_prob < 50 else
                                        (2 if median_prob < 70 else 3))
    return {'session': session, 'status': status,
            'speech_prob': median_prob, 'nihss_score': nihss,
            'wpm': None, 'speech_ratio': rows[0]['win_ratio'] if rows else 0,
            'window_start_s': round(best['start'] / SAMPLE_RATE, 1),
            'window_ratio': best['win_ratio'],
            'source_ratio': best['src_ratio'],
            'file': os.path.basename(best['file']),
            'window_probs': rows}


def main():
    extended = '--extended' in sys.argv
    first5 = '--first5' in sys.argv

    print("=" * 70)
    if extended:
        print("MODULE 2 EXTENDED TEST: ALL SESSIONS (TORGO REAL AUDIO)")
    else:
        print("MODULE 2 FULL TEST: 10 NORMAL + 5 DYSARTHRIA (TORGO REAL AUDIO)")
    print("=" * 70)
    print()

    if extended:
        normal_sessions, dys_sessions = get_all_sessions()
        print(f"Extended mode: {len(normal_sessions)} normal + {len(dys_sessions)} dys sessions")
    else:
        normal_sessions, dys_sessions = NORMAL_SESSIONS, DYS_SESSIONS
    mode = "LEGACY first-5s" if first5 else "BEST-WINDOW (VAD-guided, M2-10)"
    print(f"Window mode: {mode}")
    print()

    speech = SpeechAnalysisModule(
        vosk_model_path=VOSK_MODEL, ml_model_path=ML_MODEL, scaler_path=SCALER
    )
    if speech.ml_model is None:
        print("[ERROR] ML model khong load duoc - test khong co y nghia")
        return
    print()
    print(f"{'#':<3} {'TRUE':<7} {'SESSION':<28} {'PRED':<9} "
          f"{'PROB%':>6} {'NIHSS':>5} {'WPM':>6} {'SPK_RATIO':>9} {'WIN':>10}")
    print("-" * 90)

    results = []
    for i, session in enumerate(normal_sessions):
        r_row = run_session(speech, session, first5)
        if r_row is None:
            print(f"{i+1:<3} NORMAL {session:<28} FILE NOT FOUND")
            continue
        results.append({'true': 'NORMAL', **r_row})
        _print_row(i + 1, 'NORMAL', session, r_row)

    for i, session in enumerate(dys_sessions):
        r_row = run_session(speech, session, first5)
        if r_row is None:
            print(f"{len(normal_sessions)+i+1:<3} DYS     {session:<28} FILE NOT FOUND")
            continue
        results.append({'true': 'DYS', **r_row})
        _print_row(len(normal_sessions) + i + 1, 'DYS', session, r_row)

    # ===== METRICS =====
    normal = [r for r in results if r['true'] == 'NORMAL']
    dys = [r for r in results if r['true'] == 'DYS']

    # Binary: positive (báo động) = status WARNING/DANGER (prob >= 30)
    fp = sum(1 for r in normal if r['status'] in ('WARNING', 'DANGER'))
    tp = sum(1 for r in dys if r['status'] in ('WARNING', 'DANGER'))
    fpr = fp / len(normal) * 100 if normal else 0
    tpr = tp / len(dys) * 100 if dys else 0
    acc = (len(normal) - fp + tp) / len(results) * 100 if results else 0

    print()
    print("=" * 70)
    print("METRICS (nguong bao dong: prob >= 30%)")
    print("=" * 70)
    print(f"Confusion Matrix (binary):")
    print(f"                  Pred ALARM   Pred NORMAL")
    print(f"  Actual DYS          {tp:>4}        {len(dys)-tp:>4}")
    print(f"  Actual NORMAL       {fp:>4}        {len(normal)-fp:>4}")
    print()
    print(f"TPR  (dysarthria phat hien): {tpr:.1f}%   [target >90%] "
          f"{'PASS' if tpr > 90 else 'FAIL'}")
    print(f"FPR  (normal bao nham):      {fpr:.1f}%   [target <5%]  "
          f"{'PASS' if fpr < 5 else 'FAIL'}")
    print(f"Accuracy:                    {acc:.1f}%")

    # ===== ROC / NGUONG TOI UU (Youden's J) - giong phuong phap Module 4 =====
    probs = np.array([r['speech_prob'] for r in results])
    labels = np.array([1 if r['true'] == 'DYS' else 0 for r in results])
    best_j, best_th = -1, 30.0
    for th in np.arange(1, 100, 1.0):
        tpr_t = (labels[probs >= th] == 1).mean() if (probs >= th).any() else 0
        fpr_t = (labels[probs >= th] == 0).mean() if (probs >= th).any() else 0
        j = tpr_t - fpr_t
        if j > best_j:
            best_j, best_th = j, th
    tp_o = int(((probs >= best_th) & (labels == 1)).sum())
    fp_o = int(((probs >= best_th) & (labels == 0)).sum())
    tpr_o = tp_o / len(dys) * 100 if dys else 0
    fpr_o = fp_o / len(normal) * 100 if normal else 0
    print()
    print(f"ROC toi uu (Youden's J = {best_j:.2f}):")
    print(f"  Nguong de xuat: {best_th:.0f}%  ->  TPR {tpr_o:.1f}% | FPR {fpr_o:.1f}%  "
          f"[{'PASS' if tpr_o > 90 and fpr_o < 5 else 'CHECK'}]")
    print(f"  (Hien module dung 30%  ->  TPR {tpr:.1f}% | FPR {fpr:.1f}%)")

    # Phân bố prob trung bình theo nhóm
    if normal:
        avg_n = np.mean([r['speech_prob'] for r in normal])
        print(f"Avg prob NORMAL group: {avg_n:.1f}%")
    if dys:
        avg_d = np.mean([r['speech_prob'] for r in dys])
        print(f"Avg prob DYS group:    {avg_d:.1f}%")

    # ===== EXPORT =====
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(RESULTS_DIR, f"module2_test_{ts}.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': ts, 'model': ML_MODEL, 'extended': extended,
            'window_mode': 'first5' if first5 else 'median3_best_windows_m2_10v2',
            'threshold_alarm': 30,
            'threshold_optimal': round(float(best_th), 1),
            'TPR_optimal': round(tpr_o, 1), 'FPR_optimal': round(fpr_o, 1),
            'n_normal': len(normal), 'n_dys': len(dys),
            'TPR': round(tpr, 1), 'FPR': round(fpr, 1),
            'accuracy': round(acc, 1),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    out_csv = os.path.join(RESULTS_DIR, f"module2_test_{ts}.csv")
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write("true,session,status,prob,nihss,wpm,speech_ratio,"
                "window_start_s,window_ratio,source_ratio,file\n")
        for r in results:
            f.write(f"{r['true']},{r['session']},{r['status']},"
                    f"{r['speech_prob']:.2f},{r['nihss_score']},"
                    f"{r['wpm'] if r['wpm'] is not None else ''},{r['speech_ratio']},"
                    f"{r['window_start_s']},{r['window_ratio']},"
                    f"{r.get('source_ratio')},{r['file']}\n")
    print()
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


def _print_row(idx, true_label, session, r):
    wpm = f"{r['wpm']:>6.1f}" if r['wpm'] is not None else f"{'--':>6}"
    print(f"{idx:<3} {true_label:<7} {session.split('/')[-1]:<28} {r['status']:<9} "
          f"{r['speech_prob']:>6.1f} {r['nihss_score']:>5} "
          f"{wpm} {r['speech_ratio']:>9}"
          f" {r['window_start_s']:>7.1f}s/{r['window_ratio'] if r['window_ratio'] is not None else '--'}")


if __name__ == "__main__":
    main()
