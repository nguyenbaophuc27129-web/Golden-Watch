# -*- coding: utf-8 -*-
r"""
E-2 (NK-09) — SPEECH TIẾNG VIỆT ZERO-SHOT (L-02/M2-11)
======================================================
Protocol E2 v2.0: đội TỰ ghi âm (smoke test n nhỏ — ghi rõ trong báo cáo):
  vn_thuong\*.wav — đọc 3 câu chuẩn tiếng Việt (in sẵn, cùng câu mọi người)
  vn_liu\*.wav    — đọc CÙNG câu với lưỡi cắn/đọc nhanh méo (mô phỏng dysarthria)

Chạy model TORGO production (đóng băng, zero-shot KHÔNG tune) qua chính
SpeechAnalysisModule của app (ML MLP 48 ft + VAD + AGC) → speech_prob từng
file → AUC + accuracy + figure. Kỳ vọng trung thực: domain-shift ngôn ngữ
→ số thấp hơn TORGO là PHÁT HIỆN, kết luận "cần corpus dysarthria tiếng Việt".

Ghi âm nhanh không cần phần mềm: giữ micro gần miệng, yên tĩnh; hoặc dùng
thu âm trên web/app rồi chép file wav vào 2 thư mục trên.
Chạy: PYTHONUTF8=1 python training/eval_speech_vn_e2.py --thu vn_thuong --liu vn_liu
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, 'src')
MODELS = os.path.join(ROOT, 'models')

AUD_EXT = ('.wav', '.mp3', '.m4a', '.flac', '.ogg')


def scan_audio(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(AUD_EXT))


def load_audio(path, sr=16000):
    """Nạp audio → float32 mono 16k; mp3/m4a dùng ffmpeg của imageio-ffmpeg."""
    import soundfile as sf
    if path.lower().endswith('.wav'):
        a, file_sr = sf.read(path, dtype='float32')
    else:
        import imageio_ffmpeg
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, 'a.wav')
            subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', path,
                 '-vn', '-ac', '1', '-ar', str(sr), '-loglevel', 'error',
                 wav], check=True, capture_output=True, timeout=120)
            a, file_sr = sf.read(wav, dtype='float32')
    if a.ndim > 1:
        a = a.mean(axis=1)
    if file_sr != sr:
        import librosa
        a = librosa.resample(a, orig_sr=file_sr, target_sr=sr)
    return a.astype(np.float32)


def main():
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--thu', default='vn_thuong', help='WAV đọc chuẩn (label 0)')
    ap.add_argument('--liu', default='vn_liu', help='WAV nói líu (label 1)')
    ap.add_argument('--vosk', default=None,
                    help='đường dẫn vosk-model-small-vn (tuỳ chọn, có thì '
                         'tính thêm WPM)')
    args = ap.parse_args()

    print('== E-2 SPEECH TIẾNG VIỆT ZERO-SHOT (smoke test n nhỏ) ==')
    sys.path.insert(0, SRC_DIR)
    from detection.speech_module_v2 import SpeechAnalysisModule
    sp = SpeechAnalysisModule(
        vosk_model_path=args.vosk,
        ml_model_path=os.path.join(MODELS, 'speech_torgo_20260828_211130.pth'),
        scaler_path=os.path.join(MODELS,
                                 'speech_torgo_20260828_211130_scaler.pkl'))

    rows = []
    for folder, label, tag in ((args.thu, 0, 'vn_thuong'),
                               (args.liu, 1, 'vn_liu')):
        files = scan_audio(folder)
        print(f'{tag}: {len(files)} file')
        for fp in files:
            try:
                audio = load_audio(fp)
            except Exception as e:
                print(f'  {os.path.basename(fp)}: lỗi đọc ({e})')
                continue
            r = sp.predict_dysarthria(audio, len(audio) / 16000)
            prob = float(r.get('speech_prob', 0) or 0)
            status = r.get('status', 'NO_SPEECH')
            if status == 'NO_SPEECH':
                print(f'  {os.path.basename(fp)}: NO_SPEECH (im lặng/nhiễu?)')
                continue
            rows.append({'file': os.path.basename(fp), 'cond': tag,
                         'label': label, 'prob': prob,
                         'wpm': r.get('metrics', {}).get('wpm'),
                         'dur_s': round(len(audio) / 16000, 1)})
            print(f"  {os.path.basename(fp)}: prob {prob:.1f} "
                  f"(WPM {r.get('metrics', {}).get('wpm')})")

    if len({r['label'] for r in rows}) < 2:
        print('\n[!] Cần ít nhất 1 file mỗi thư mục để eval. Thoát.')
        return

    y = np.array([r['label'] for r in rows])
    p = np.array([r['prob'] for r in rows])
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else None
    # ngưỡng "nửa khoảng": prob nằm ở 0–1 hay 0–100 tuỳ đầu ra module
    acc = float((((p >= 0.5) if p.max() <= 1 else (p >= 50)).astype(int) == y)
                .mean())

    out_dir = os.path.join(
        ROOT, 'test_results',
        f'speech_vn_e2_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)
    res = {'protocol': 'E2 speech VN zero-shot TORGO (smoke test n nhỏ, '
                       'model đóng băng KHÔNG tune)',
           'n_files': len(rows), 'n_thuong': int((y == 0).sum()),
           'n_liu': int((y == 1).sum()),
           'auc': None if auc is None else round(auc, 4),
           'accuracy@nguong_nua_khoang': round(acc, 4),
           'rows': rows}
    with open(os.path.join(out_dir, 'summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=170)
    cols = ['#1a6faf' if r['label'] == 0 else '#c0392b' for r in rows]
    names = [r['file'][:12] for r in rows]
    ax.bar(range(len(rows)), p, color=cols)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    lo, hi = (0, 100) if p.max() > 1 else (0, 1)
    ax.axhline((lo + hi) / 2, color='k', ls='--', lw=1.2,
               label='ngưỡng nửa khoảng')
    ax.set_ylim(lo, hi)
    ax.set_ylabel('speech_prob (TORGO zero-shot)')
    ax.set_title(f"Tiếng Việt — đọc chuẩn vs nói líu "
                 f"(AUC {'—' if auc is None else f'{auc:.3f}'}, n={len(rows)})")
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'speech_vn_e2.png'))

    print(f"\nAUC: {'—' if auc is None else f'{auc:.4f}'} · accuracy "
          f"{acc*100:.1f}% · n={len(rows)} (thương {int((y==0).sum)} / "
          f"liu {int((y==1).sum)})")
    print('Kỳ vọng trung thực: thấp hơn TORGO 0.62 người-level = phát hiện '
          'domain-shift ngôn ngữ.')
    print(f'Đã lưu: {out_dir} (ghi NK-10 vào nhat ky.md)')


if __name__ == '__main__':
    main()
