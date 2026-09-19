# -*- coding: utf-8 -*-
"""
SYS-18 — NIHSS VIDEO TOOLKIT (template + score-video + eval)
=============================================================
Bộ công cụ hoàn chỉnh cho Protocol SYS-18 (50 video NIHSS công khai):

  1) template    — tạo CSV chấm tay (2 chấm viên độc lập → consensus).
  2) score-video — chạy 4 module THẬT của hệ thống (face ML v3, arm YOLO,
                   gait YOLO, speech MLP) trên từng file video → điền
                   cột machine_* vào CSV. Audio tách bằng ffmpeg của
                   imageio-ffmpeg (không cần cài ffmpeg hệ thống).
  3) eval        — thống kê máy-vs-người: MAE, RMSE, r², weighted-κ
                   (quadratic), Bland–Altman + inter-rater κ + 3 figure.

Chạy:
  PYTHONUTF8=1 python training/nihss_video_toolkit.py template
  PYTHONUTF8=1 python training/nihss_video_toolkit.py score-video video1.mp4 ...
  PYTHONUTF8=1 python training/nihss_video_toolkit.py eval --csv NIHSS_SY18_bang_cham_video.csv
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'models')
SRC_DIR = os.path.join(ROOT, 'src')
DEFAULT_CSV = os.path.join(ROOT, 'NIHSS_SY18_bang_cham_video.csv')

ITEMS = ['item4_facial_palsy', 'item5_motor_arm',
         'item6_motor_leg', 'item10_dysarthria']
ITEM_MAX = {'item4_facial_palsy': 3, 'item5_motor_arm': 4,
            'item6_motor_leg': 4, 'item10_dysarthria': 2}
COLS = (['video_id', 'url', 'ref_total']
        + [f'rater1_item{i}' for i in (4, 5, 6, 10)]
        + [f'rater2_item{i}' for i in (4, 5, 6, 10)]
        + [f'agreed_item{i}' for i in (4, 5, 6, 10)]
        + [f'machine_item{i}' for i in (4, 5, 6, 10)]
        + ['machine_total', 'notes'])


# ======================================================================
# 1) TEMPLATE
# ======================================================================
def make_template(out_path):
    if os.path.exists(out_path):
        print(f'Đã có, không ghi đè: {out_path}')
        return out_path
    pd.DataFrame(columns=COLS).to_csv(out_path, index=False,
                                      encoding='utf-8-sig')
    print(f'Đã tạo template: {out_path}')
    print('Cách điền:')
    print('  video_id  : tên ngắn (vd nihss_demo_01) — trùng tên file')
    print('              khi chạy score-video để máy điền cột machine_*')
    print('  url       : link YouTube nguồn (công khai)')
    print('  ref_total : NIHSS tổng do bác sĩ công bố trong video')
    print('  rater1_*, rater2_* : 2 chấm viên ĐỘC LẬP chấm từng item')
    print('  agreed_*  : điểm đồng thuận sau thảo luận (so máy với cột này)')
    print('  Điểm item4 0–3 · item5/item6 0–4 · item10 0–2 (tổng max 13)')
    return out_path


# ======================================================================
# 2) SCORE-VIDEO — chạy 4 module thật
# ======================================================================
def init_detectors():
    sys.path.insert(0, SRC_DIR)
    from detection.face_module_v7 import FaceAsymmetryDetector
    from detection.face_ml_v3 import FaceMLV3
    from detection.arm_module import ArmWeaknessDetector
    from detection.gait_module import GaitPoseDetector
    from detection.speech_module_v2 import SpeechAnalysisModule

    face = FaceAsymmetryDetector(
        model_path=os.path.join(MODELS, 'face_landmarker_v2.task'))
    face.ml_model = FaceMLV3.load_latest(MODELS)      # SYS-29
    arm = ArmWeaknessDetector(
        pose_model_path=os.path.join(SRC_DIR, 'yolov8n-pose.pt'),
        ml_model_path=os.path.join(MODELS, 'arm_weakness_20260830_200657.pth'),
        scaler_path=os.path.join(MODELS,
                                 'arm_weakness_20260830_200657_scaler.pkl'))
    gait = GaitPoseDetector(model_path=os.path.join(SRC_DIR,
                                                    'yolov8n-pose.pt'))
    speech = SpeechAnalysisModule(
        ml_model_path=os.path.join(MODELS, 'speech_torgo_20260828_211130.pth'),
        scaler_path=os.path.join(MODELS,
                                 'speech_torgo_20260828_211130_scaler.pkl'))
    return face, arm, gait, speech


def extract_audio_wav(video_path, out_wav):
    """Tách audio 16k mono bằng ffmpeg của imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        print('  [!] thiếu imageio-ffmpeg → pip install imageio-ffmpeg')
        return None
    cmd = [exe, '-y', '-i', video_path, '-vn', '-ac', '1', '-ar', '16000',
           '-loglevel', 'error', out_wav]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    except Exception as e:
        print(f'  [!] ffmpeg lỗi: {e}')
        return None
    ok = os.path.exists(out_wav) and os.path.getsize(out_wav) > 44
    return out_wav if ok else None


def agg_prob(vals):
    """Median prob các frame hợp lệ; None nếu không có frame hợp lệ."""
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def score_one_video(path, detectors, sample_hz=2.0):
    import cv2
    face, arm, gait, speech = detectors
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f'  [!] không mở được video: {path}')
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_hz)))
    f_scores, a_probs, g_probs, n_frames = [], [], [], 0
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            n_frames += 1
            r = face.process_frame(rgb)
            if r.get('status') not in ('NO_FACE', 'PARTIAL_FACE', 'ERROR'):
                f_scores.append(float(r.get('score', 0) or 0))
            ra = arm.detect_arm_weakness(rgb)
            if ra.get('status') not in ('NO_PERSON', 'NO_POSE'):
                a_probs.append(float(ra.get('arm_prob', 0) or 0))
            rg = gait.detect_gait_from_frame(rgb)
            if rg.get('status') not in ('NO_PERSON', 'NO_POSE'):
                g_probs.append(float(rg.get('gait_prob', 0) or 0))
        i += 1
    cap.release()
    dur = i / fps

    mods = {}
    fs = agg_prob(f_scores)
    mods['face'] = ({'score': fs, 'status': 'OK'} if fs is not None
                    else {'score': 0, 'status': 'NO_FACE'})
    ap = agg_prob(a_probs)
    mods['arm'] = ({'arm_prob': ap, 'status': 'OK'} if ap is not None
                   else {'arm_prob': 0, 'status': 'NO_PERSON'})
    gp = agg_prob(g_probs)
    mods['gait'] = ({'gait_prob': gp, 'status': 'OK'} if gp is not None
                    else {'gait_prob': 0, 'status': 'NO_PERSON'})

    with tempfile.TemporaryDirectory() as td:
        wav = extract_audio_wav(path, os.path.join(td, 'a.wav'))
        if wav:
            import soundfile as sf
            audio, sr = sf.read(wav, dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            rs = speech.predict_dysarthria(audio, len(audio) / sr)
            mods['speech'] = ({'speech_prob': float(rs.get('speech_prob', 0)),
                               'status': rs.get('status', 'NO_SPEECH')}
                              if rs.get('status') != 'NO_SPEECH'
                              else {'speech_prob': 0, 'status': 'NO_SPEECH'})
        else:
            mods['speech'] = {'speech_prob': 0, 'status': 'NO_DATA'}

    from fusion.nihss_estimator import estimate_nihss
    est = estimate_nihss(mods)
    items = est['items']
    out = {f'machine_item{i}': np.nan for i in (4, 5, 6, 10)}
    name_map = {'item4_facial_palsy': 'machine_item4',
                'item5_motor_arm': 'machine_item5',
                'item6_motor_leg': 'machine_item6',
                'item10_dysarthria': 'machine_item10'}
    for it, col in name_map.items():
        if it in items:
            out[col] = int(items[it])
    out['machine_total'] = est['total']
    print(f'  {os.path.basename(path)}: {n_frames} frame, {dur:.0f}s → '
          f"face {fs if fs is not None else '—'} · "
          f"arm {ap if ap is not None else '—'} · "
          f"gait {gp if gp is not None else '—'} · "
          f"total {est['total']}/13 (missing {est['items_missing']})")
    return out


def run_score_video(videos, csv_path):
    detectors = init_detectors()
    df = (pd.read_csv(csv_path, encoding='utf-8-sig') if os.path.exists(csv_path)
          else pd.DataFrame(columns=COLS))
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    for vp in videos:
        vid = os.path.splitext(os.path.basename(vp))[0]
        print(f'Đang chấm: {vid}')
        res = score_one_video(vp, detectors)
        if res is None:
            continue
        m = df['video_id'].astype(str) == vid
        if not m.any():
            row = {'video_id': vid, 'url': '', 'ref_total': np.nan,
                   'notes': f'auto-scored {datetime.now():%Y-%m-%d %H:%M}'}
            row.update(res)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            for k, v in res.items():
                df.loc[m, k] = v
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'Đã cập nhật: {csv_path}')


# ======================================================================
# 3) EVAL — thống kê máy vs người
# ======================================================================
def w_kappa(a, b, kmax):
    from sklearn.metrics import cohen_kappa_score
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return None
    try:
        return float(cohen_kappa_score(
            a, b, weights='quadratic',
            labels=list(range(kmax + 1))))
    except Exception:
        return None


def run_eval(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    n = len(df)
    if n == 0:
        print('CSV trống — hãy điền điểm chấm tay trước (mode template).')
        return

    out_dir = os.path.join(
        ROOT, 'test_results',
        f'nihss_video_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)

    # --- kiểm tra biên độ: giá trị ngoài khoảng → NaN thật sự ---
    kmap = {4: 3, 5: 4, 6: 4, 10: 2}
    for i, kmax in kmap.items():
        for pre in ('rater1', 'rater2', 'agreed', 'machine'):
            col = f'{pre}_item{i}'
            v = pd.to_numeric(df[col], errors='coerce')
            bad = v.notna() & ((v < 0) | (v > kmax))
            if bad.any():
                print(f'[!] {col}: {int(bad.sum())} giá trị ngoài 0–{kmax}'
                      ' → bỏ (NaN)')
                df.loc[bad, col] = np.nan

    res = {'protocol': 'SYS-18 NIHSS video toolkit', 'n_videos': n,
           'csv': os.path.basename(csv_path), 'items': {}, 'total': {},
           'inter_rater': {}}

    # --- per item: máy vs consensus + inter-rater ---
    rows = []
    for i, kmax in zip((4, 5, 6, 10), (3, 4, 4, 2)):
        ag = pd.to_numeric(df[f'agreed_item{i}'], errors='coerce')
        mc = pd.to_numeric(df[f'machine_item{i}'], errors='coerce')
        r1 = pd.to_numeric(df[f'rater1_item{i}'], errors='coerce')
        r2 = pd.to_numeric(df[f'rater2_item{i}'], errors='coerce')
        sel = ag.notna() & mc.notna()
        entry = {'n_machine_vs_agreed': int(sel.sum())}
        if sel.sum() >= 2:
            d = (mc[sel] - ag[sel]).astype(float)
            r = float(np.corrcoef(mc[sel], ag[sel])[0, 1]) \
                if np.std(ag[sel]) > 0 and np.std(mc[sel]) > 0 else None
            entry.update({
                'mae': round(float(d.abs().mean()), 3),
                'rmse': round(float(np.sqrt((d ** 2).mean())), 3),
                'bias': round(float(d.mean()), 3),
                'pearson_r': None if r is None else round(r, 3),
                'r2': None if r is None else round(r * r, 3),
                'weighted_kappa': w_kappa(mc[sel], ag[sel], kmax)})
        both = r1.notna() & r2.notna()
        if both.sum() >= 2:
            k = w_kappa(r1[both], r2[both], kmax)
            res['inter_rater'][f'item{i}'] = \
                None if k is None else round(k, 3)
            entry['n_inter_rater'] = int(both.sum())
        res['items'][f'item{i}'] = entry
        rows.append((f'item{i}', entry))

    # --- total ---
    ag_tot = sum(pd.to_numeric(df[f'agreed_item{i}'], errors='coerce')
                 for i in (4, 5, 6, 10))
    mc_tot = pd.to_numeric(df['machine_total'], errors='coerce')
    ref_tot = pd.to_numeric(df['ref_total'], errors='coerce')
    sel = ag_tot.notna() & mc_tot.notna()
    if sel.sum() >= 2:
        d = (mc_tot[sel] - ag_tot[sel]).astype(float)
        r = float(np.corrcoef(mc_tot[sel], ag_tot[sel])[0, 1]) \
            if np.std(ag_tot[sel]) > 0 else None
        bias, sd = float(d.mean()), float(d.std(ddof=1))
        res['total'] = {'n': int(sel.sum()), 'mae': round(float(d.abs().mean()), 3),
                        'rmse': round(float(np.sqrt((d ** 2).mean())), 3),
                        'bias': round(bias, 3),
                        'loa': [round(bias - 1.96 * sd, 3),
                                round(bias + 1.96 * sd, 3)],
                        'pearson_r': None if r is None else round(r, 3),
                        'r2': None if r is None else round(r * r, 3)}
        sel2 = ref_tot.notna() & ag_tot.notna()
        if sel2.sum() >= 2:
            d2 = (ag_tot[sel2] - ref_tot[sel2]).astype(float)
            res['total']['human_vs_ref_mae'] = round(float(d2.abs().mean()), 3)

    # --- FIGURES ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if res['total']:
        t = res['total']
        x = ag_tot[sel].astype(float)
        y = mc_tot[sel].astype(float)
        m_ = (x + y) / 2
        d_ = y - x
        fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=170)
        ax.scatter(m_, d_, s=46, color='#1a6faf', edgecolor='k', zorder=3)
        ax.axhline(t['bias'], color='#c0392b', lw=1.8,
                   label=f"bias {t['bias']:+.2f}")
        ax.axhline(t['loa'][0], color='#e67e22', ls='--', lw=1.4,
                   label=f"LoA dưới {t['loa'][0]:.2f}")
        ax.axhline(t['loa'][1], color='#e67e22', ls='--', lw=1.4,
                   label=f"LoA trên {t['loa'][1]:.2f}")
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_xlabel('Trung bình (máy, consensus) — NIHSS 4 item /13')
        ax.set_ylabel('Chênh (máy − consensus)')
        ax.set_title(f"Bland–Altman máy vs người (n={t['n']})")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'bland_altman_total.png'))

        fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=170)
        lo = float(min(x.min(), y.min())) - 0.5
        hi = float(max(x.max(), y.max())) + 0.5
        ax.scatter(x, y, s=52, color='#1a6faf', edgecolor='k', zorder=3)
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='y = x (hoàn hảo)')
        if t.get('r2') is not None:
            ax.set_title(f"Máy vs consensus — r²={t['r2']}, "
                         f"MAE={t['mae']} điểm")
        ax.set_xlabel('NIHSS consensus (2 chấm viên)')
        ax.set_ylabel('NIHSS máy (4 module)')
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'scatter_total.png'))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=170)
    names = [r[0] for r in rows]
    maes = [r[1].get('mae') if r[1].get('mae') is not None else np.nan
            for r in rows]
    kps = [r[1].get('weighted_kappa') for r in rows]
    axes[0].bar(names, maes, color='#1a6faf')
    axes[0].set_title('MAE từng item (máy vs consensus)')
    axes[0].set_ylabel('MAE (điểm)')
    axes[0].grid(axis='y', alpha=0.3)
    kv = [k if k is not None else np.nan for k in kps]
    axes[1].bar(names, kv, color='#27ae60')
    for j, k in enumerate(kv):
        if not np.isnan(k):
            axes[1].text(j, k + 0.02, f'{k:.2f}', ha='center', fontsize=9)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title('Weighted-κ (quadratic) từng item')
    axes[1].grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'per_item_stats.png'))

    with open(os.path.join(out_dir, 'summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    print(f"\n== KẾT QUẢ SYS-18 (n={n} video) ==")
    for name, e in rows:
        if not e:
            continue
        k = e.get('weighted_kappa')
        print(f"  {name}: MAE {e.get('mae')} · RMSE {e.get('rmse')} · "
              f"r² {e.get('r2')} · wκ {None if k is None else round(k, 3)} "
              f"(n={e.get('n_machine_vs_agreed')})")
    if res['total']:
        t = res['total']
        print(f"  TOTAL: MAE {t['mae']} · RMSE {t['rmse']} · bias "
              f"{t['bias']} · LoA {t['loa']} · r² {t.get('r2')}")
        if 'human_vs_ref_mae' in res['total']:
            print(f"  Người vs NIHSS công bố: MAE "
                  f"{res['total']['human_vs_ref_mae']}")
    irr = {k: v for k, v in res['inter_rater'].items() if v is not None}
    if irr:
        print(f"  Inter-rater wκ: {irr}")
    print(f'Đã lưu: {out_dir}')
    return res


def main():
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('template')
    t.add_argument('--out', default=DEFAULT_CSV)
    s = sub.add_parser('score-video')
    s.add_argument('videos', nargs='+')
    s.add_argument('--csv', default=DEFAULT_CSV)
    e = sub.add_parser('eval')
    e.add_argument('--csv', default=DEFAULT_CSV)
    args = ap.parse_args()

    if args.cmd == 'template':
        make_template(args.out)
    elif args.cmd == 'score-video':
        run_score_video(args.videos, args.csv)
    elif args.cmd == 'eval':
        run_eval(args.csv)


if __name__ == '__main__':
    main()
