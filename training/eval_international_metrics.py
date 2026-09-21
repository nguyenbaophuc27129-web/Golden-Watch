# -*- coding: utf-8 -*-
"""NK-35: ĐO BỘ THANG ĐO QUỐC TẾ (Mục III gợi ý GVHD) — KHÔNG train lại.

Nguyên tắc: KHÔNG đụng model/ngưỡng đã khóa (NK-12). Mọi số đo trên dữ liệu
THẬT đã có (đều là dataset quốc tế ẩn danh: Kaggle stroke face, PhysioNet
gait, TORGO speech):

 1. Face v3 (model chính thức) — OOF GroupKFold(5) theo block, C=0.03
    (đúng protocol NK-24): Sens/Spec/FPR/FNR, AUPRC, Brier, ECE (10 bins).
 2. Speech LOSO NGƯỜI-THẬT (oof_predictions.csv NK-03 — số trung thực):
    Sens/Spec/FPR/FNR @0.5, AUPRC, Brier, ECE.
 3. Gait v2 — trích từ metrics_summary.json (LOSO đã chốt M4-07).
 4. Fusion 4 mức — MÔ PHỎNG (seed 42) chạy qua FusionEngine THẬT
    (DEFAULT_WEIGHTS + ngưỡng 30/50/70) → confusion matrix 4x4
    NORMAL/MONITOR/WARNING/EMERGENCY + FPR/FNR hệ thống.
 5. Latency to Alarm — trích số replay early_warning (NK-11) → biểu đồ.
 6. System: FPS từng nhánh đo trực tiếp trên máy (MediaPipe + YOLO pose)
    + MTTD theo ngân sách thiết kế (radar 45s confirm + chu kỳ 2.2s).

Output: test_results/international_metrics_<ts>/ (summary.json + CSV + PNG)
Chạy:  PYTHONUTF8=1 python training/eval_international_metrics.py
"""
import json
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, 'test_results',
                   'international_metrics_%s' % time.strftime('%Y%m%d_%H%M%S'))
os.makedirs(OUT, exist_ok=True)
SEED = 42
summary = {'purpose': 'NK-35 — Thang đo quốc tế Mục III (Sens/Spec/FPR/FNR, '
                      'AUPRC, Brier/ECE, Latency, Confusion 4 mức, FPS/MTTD)',
           'protocol_freeze': 'NK-12: model + ngưỡng KHÔNG đổi, chỉ đo'}

# ------------------------------------------------------------------
# 1) FACE v3 — OOF GroupKFold(5) theo block từ cache đặc trưng
# ------------------------------------------------------------------
CACHE = os.path.join(ROOT, 'test_results', '_face_v3_features_cache.npz')
d = np.load(CACHE, allow_pickle=True)
X, y, blocks = d['X'], d['y'].astype(int), d['blocks']
gkf = GroupKFold(5)
oof_p = np.zeros(len(y))
for tr, te in gkf.split(X, y, groups=blocks):
    # chuẩn hóa trong fold (không nhìn test)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    clf = LogisticRegression(C=0.03, max_iter=2000, random_state=SEED)
    clf.fit((X[tr] - mu) / sd, y[tr])
    oof_p[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]

THR_FACE = 0.298  # ngưỡng Youden công bố của artifact chính thức
pred = (oof_p >= THR_FACE).astype(int)
tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())


def cls_metrics(tp, fp, fn, tn):
    sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
    return {'n': tp + fp + fn + tn, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'sensitivity': round(sens, 4),
            'specificity': round(spec, 4),
            'FPR': round(1 - spec, 4), 'FNR': round(1 - sens, 4),
            'precision': round(tp / max(1, tp + fp), 4),
            'f1': round(2 * tp / max(1, 2 * tp + fp + fn), 4)}


def ece_score(p, y, bins=10):
    ece, rows = 0.0, []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if m.sum() == 0:
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.sum() / len(p) * abs(conf - acc)
        rows.append({'bin': i, 'n': int(m.sum()),
                     'conf': round(conf, 4), 'acc': round(acc, 4)})
    return round(ece, 4), rows


face = cls_metrics(tp, fp, fn, tn)
face.update({'auc_oof': round(float(roc_auc_score(y, oof_p)), 4),
             'auprc': round(float(average_precision_score(y, oof_p)), 4),
             'auprc_baseline_prevalence': round(float(y.mean()), 4),
             'brier': round(float(brier_score_loss(y, oof_p)), 4),
             'ece_10bin': ece_score(oof_p, y)[0],
             'ece_published_nk24_inner_c': 0.021,
             'brier_published_nk24': 0.0783,
             'threshold': THR_FACE,
             'protocol': 'OOF GroupKFold(5) theo block, C=0.03, seed 42 — '
                         'đúng protocol NK-24; thr = Youden artifact 0.298; '
                         'ECE công bố NK-24 (2.1%) dùng C chọn inner-OOF '
                         'mỗi fold — số đo nhanh ở đây tham chiếu'})
summary['face_v3'] = face
print('[FACE v3]', json.dumps(face, ensure_ascii=False))

# ------------------------------------------------------------------
# 2) SPEECH — LOSO NGƯỜI-THẬT (số trung thực NK-03)
# ------------------------------------------------------------------
import csv
ps, ys = [], []
with open(os.path.join(ROOT, 'test_results', 'speech_speaker_loso_'
                       '20260910_214834', 'oof_predictions.csv'),
          encoding='utf-8') as f:
    for row in csv.DictReader(f):
        ps.append(float(row['p_logreg'])); ys.append(int(row['label']))
ps, ys = np.array(ps), np.array(ys)
pred_s = (ps >= 0.5).astype(int)
tsp = int(((pred_s == 1) & (ys == 1)).sum())
fsp = int(((pred_s == 1) & (ys == 0)).sum())
fsn = int(((pred_s == 0) & (ys == 1)).sum())
tsn = int(((pred_s == 0) & (ys == 0)).sum())
speech = cls_metrics(tsp, fsp, fsn, tsn)
speech.update({'auc_loso': round(float(roc_auc_score(ys, ps)), 4),
               'auprc': round(float(average_precision_score(ys, ps)), 4),
               'auprc_baseline_prevalence': round(float(ys.mean()), 4),
               'brier': round(float(brier_score_loss(ys, ps)), 4),
               'ece_10bin': ece_score(ps, ys)[0],
               'threshold': 0.5,
               'protocol': 'LOSO theo NGƯỜI THẬT 15 subject (khử leakage '
                           'session NK-03) — số trung thực, KHÔNG dùng 0.992'})
summary['speech_loso'] = speech
print('[SPEECH LOSO]', json.dumps(speech, ensure_ascii=False))

# ------------------------------------------------------------------
# 3) GAIT — số đã chốt trong metrics_pack (không tính lại)
# ------------------------------------------------------------------
pack = json.load(open(os.path.join(ROOT, 'test_results',
                                   'metrics_pack_20260907_223217',
                                   'metrics_summary.json'), encoding='utf-8'))
g = next(r for r in pack['results'] if r['module'] == 'gait_loso_v2')
summary['gait_v2'] = {'sensitivity': g['recall_sens'],
                      'specificity': g['specificity'], 'FPR': round(1 - g['specificity'], 4),
                      'FNR': round(1 - g['recall_sens'], 4),
                      'auc_loso': g['auc_roc'], 'f1': g['f1'],
                      'protocol': 'LOSO PhysioNet 15 subject — metrics_pack 07/09'}
print('[GAIT v2]', json.dumps(summary['gait_v2'], ensure_ascii=False))

# ------------------------------------------------------------------
# 4) FUSION 4 MỨC — mô phỏng seed 42 qua FusionEngine THẬT
# ------------------------------------------------------------------
from fusion.fusion_engine import FusionEngine   # noqa: E402

rng = np.random.default_rng(SEED)
# dict đúng KEY adapter fusion: face->score, speech->speech_prob,
# arm->arm_prob, gait->gait_prob, radar->fall_prob
PKEY = {'face': 'score', 'speech': 'speech_prob', 'arm': 'arm_prob',
        'gait': 'gait_prob', 'radar': 'fall_prob'}
SCEN = {   # phân bố prob mỗi module theo lớp TRUTH (hiệu chỉnh để trung
    # bình lớp rơi đúng dải ngưỡng 30/50/70 với trọng số thật)
    'NORMAL':    dict(face=(5, 20),  speech=(0, 10),  arm=(0, 15),
                      gait=(0, 35),  radar=(0, 10)),
    'MONITOR':   dict(face=(25, 50), speech=(10, 40), arm=(15, 45),
                      gait=(20, 60), radar=(10, 40)),
    'WARNING':   dict(face=(55, 85), speech=(40, 70), arm=(50, 80),
                      gait=(40, 80), radar=(30, 60)),
    'EMERGENCY': dict(face=(80, 98), speech=(70, 98), arm=(75, 98),
                      gait=(70, 98), radar=(60, 90)),
}
engine = FusionEngine()
truth, pred_lvl = [], []
for lvl, dist in SCEN.items():
    for _ in range(200):
        mods = {}
        for k, (lo, hi) in dist.items():
            prob = float(rng.uniform(lo, hi))
            if k == 'speech' and prob < 5:      # im lặng → NO_SPEECH bị loại
                mods[k] = {'status': 'NO_SPEECH', PKEY[k]: prob}
                continue
            risk = ('NORMAL' if prob < 30 else 'MONITOR' if prob < 50
                    else 'WARNING' if prob < 70 else 'EMERGENCY')
            mods[k] = {'status': risk, PKEY[k]: prob}
        try:
            out = engine.fuse(mods)
            pred_lvl.append(out.get('risk_level', 'NORMAL'))
            truth.append(lvl)
        except Exception as e:
            print('[FUSION warn]', repr(e))
LV = ['NORMAL', 'MONITOR', 'WARNING', 'EMERGENCY']
cm = confusion_matrix(truth, pred_lvl, labels=LV)
alert_true = np.array([LV.index(t) >= 2 for t in truth])
alert_pred = np.array([LV.index(p) >= 2 for p in pred_lvl])
stp = int((alert_true & alert_pred).sum()); sfp = int((~alert_true & alert_pred).sum())
sfn = int((alert_true & ~alert_pred).sum()); stn = int((~alert_true & ~alert_pred).sum())
fusion = cls_metrics(stp, sfp, sfn, stn)
fusion.update({'confusion_4level': cm.tolist(), 'labels': LV,
               'n_per_class': 200, 'seed': SEED,
               'protocol': 'MÔ PHỎNG seed 42 chạy qua FusionEngine THẬT '
                           '(weights face .20 speech .20 arm .30 gait .15 '
                           'radar .15; ngưỡng 30/50/70) — kiểm tra định '
                           'tuyến tầng fusion, KHÔNG phải khả năng phát hiện'})
summary['fusion_sim'] = fusion
print('[FUSION SIM]', json.dumps({k: v for k, v in fusion.items()
                                  if k != 'confusion_4level'},
                                 ensure_ascii=False))

# ------------------------------------------------------------------
# 5) LATENCY TO ALARM — số replay early_warning NK-11
# ------------------------------------------------------------------
ew = json.load(open(os.path.join(ROOT, 'test_results',
                                 'early_warning_20260912_124947',
                                 'summary.json'), encoding='utf-8'))
lat = ew['latency']
summary['latency_to_alarm'] = {
    'protocol': ew.get('protocol', '') + ' — FAR 0/24h cả 2 hệ',
    'ramps': {k: {'old_min': (v['old'] or {}).get('median_min'),
                  'new_min': (v['new'] or {}).get('median_min'),
                  'old_fired': v.get('old_ever_fired'),
                  'new_fired': v.get('new_ever_fired')} for k, v in lat.items()},
    'headline': 'ramp +18/30ph DƯỚI ngưỡng: hệ cũ 0/20 KHÔNG BAO GIỜ vs '
                'mới 20/20 @21.6ph; giảm 45–50% độ trễ ở 3 ramp trên ngưỡng'}
print('[LATENCY]', summary['latency_to_alarm']['headline'])

# ------------------------------------------------------------------
# 6) SYSTEM — FPS từng nhánh (đo trực tiếp) + MTTD ngân sách thiết kế
# ------------------------------------------------------------------
sys_perf = {'protocol': 'Đo trực tiếp trên máy RTX 3050 (khung 1280x720), '
                        '30 khung; MTTD = ngân sách thiết kế từ spec radar'}
try:
    img_path = os.path.join(ROOT, 'exports', 'nk28_mesh_demo.png')
    import cv2
    frame = cv2.imread(img_path)
    if frame is None:
        cap = cv2.VideoCapture(0); ok, frame = cap.read(); cap.release()
        frame = cv2.flip(frame, 1) if ok else np.zeros((720, 1280, 3), np.uint8)
    if frame.shape[1] != 1280:
        frame = cv2.resize(frame, (1280, 720))

    t0 = time.perf_counter(); N = 20
    import torch
    from ultralytics import YOLO
    ym = YOLO(os.path.join(ROOT, 'src', 'yolov8n-pose.pt'))
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    for _ in range(3):   # warm-up CUDA/NMS
        ym.predict(frame, verbose=False, conf=0.10, imgsz=640, device=dev)
    for _ in range(N):
        ym.predict(frame, verbose=False, conf=0.10, imgsz=640, device=dev)
    yolo_ms = (time.perf_counter() - t0) / N * 1000

    from detection.face_module_v7 import FaceAsymmetryDetector
    fm = FaceAsymmetryDetector()
    fm.process_frame(frame)  # warm-up
    t0 = time.perf_counter()
    for _ in range(N):
        fm.process_frame(frame)
    face_ms = (time.perf_counter() - t0) / N * 1000

    sys_perf.update({'yolo_pose_ms_per_frame': round(yolo_ms, 1),
                     'face_landmarker_ms_per_frame': round(face_ms, 1),
                     'camera_fps': 30,
                     'yolo_device': dev,
                     'note': 'chu kỳ phân tích đầy đủ đã audit NK-26: '
                             '80.9ms — 4 luồng song song không nghẽn; '
                             'bench đơn lẻ ở đây chỉ tham chiếu'})
except Exception as e:
    sys_perf['error'] = repr(e)
    sys_perf['note'] = 'dùng số audit NK-26: chu kỳ đầy đủ 80.9ms'
# MTTD: radar fall cần confirm trong cửa sổ spec (45s) + chu kỳ phân tích 2.2s
sys_perf['mttd_budget_s'] = {'radar_fall_confirm_window_s': 45,
                             'analysis_cycle_s': 2.2,
                             'alert_push_s': 0.2,
                             'total_worst_case_s': 47.4,
                             'note': 'ngã đã confirm → cảnh báo ở chu kỳ kế '
                                     'tiếp (~2.4s) — smoke NK-32'}
summary['system'] = sys_perf
print('[SYSTEM]', json.dumps(sys_perf, ensure_ascii=False))

# ------------------------------------------------------------------
# BẢNG MỤC TIÊU MDR + CSV
# ------------------------------------------------------------------
targets = {'FPR_max': 0.05, 'FNR_max': 0.01,
           'note': 'MDR Class IIa/IIb — mục tiêu GVHD đề'}
summary['mdr_targets'] = targets
rows = []
for mod in ('face_v3', 'speech_loso', 'gait_v2'):
    m = summary[mod]
    rows.append([mod, m['sensitivity'], m['specificity'], m['FPR'], m['FNR'],
                 m.get('auprc', m.get('auc_loso')), m.get('brier', ''),
                 m.get('ece_10bin', ''), m['protocol']])
rows.append(['fusion_sim', fusion['sensitivity'], fusion['specificity'],
             fusion['FPR'], fusion['FNR'], '', '', '', fusion['protocol']])
with open(os.path.join(OUT, 'bang_so_lieu_muc3.csv'), 'w', newline='',
          encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['module', 'sensitivity', 'specificity', 'FPR', 'FNR',
                'AUPRC/AUC', 'Brier', 'ECE_10bin', 'protocol'])
    w.writerows(rows)

json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)

# ------------------------------------------------------------------
# BIỂU ĐỒ
# ------------------------------------------------------------------
plt.rcParams['font.size'] = 9

# (1) PR curves — AUPRC
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for a, (name, p, yy) in zip(ax, [
        ('Face v3 (OOF block)', oof_p, y),
        ('Speech LOSO nguoi-that', ps, ys)]):
    prec, rec, _ = precision_recall_curve(yy, p)
    ap = average_precision_score(yy, p)
    a.step(rec, prec, where='post', color='#1f77b4', lw=2)
    a.axhline(yy.mean(), color='gray', ls='--', lw=1,
              label=f'baseline ( prevalence={yy.mean():.2f})')
    a.set_xlabel('Recall'); a.set_ylabel('Precision')
    a.set_title(f'{name}\nAUPRC = {ap:.4f}')
    a.legend(); a.set_ylim(0, 1.02); a.grid(alpha=.3)
fig.suptitle('AUPRC — du lieu mat can bang (imbalanced)', y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'pr_curves_auprc.png'),
                                dpi=150, bbox_inches='tight'); plt.close(fig)

# (2) Confusion matrix 4 mức fusion
fig, ax = plt.subplots(figsize=(5.4, 4.6))
cmn = cm / cm.sum(1, keepdims=True)
im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{cm[i, j]}\n({cmn[i, j]*100:.0f}%)',
                ha='center', va='center', fontsize=9,
                color='white' if cmn[i, j] > .5 else 'black')
ax.set_xticks(range(4), LV, rotation=20); ax.set_yticks(range(4), LV)
ax.set_xlabel('Dự đoán FusionEngine'); ax.set_ylabel('Lớp mô phỏng (truth)')
ax.set_title('Fusion 4 mức — mô phỏng seed 42 qua FusionEngine THẬT\n'
             f'Sens {fusion["sensitivity"]*100:.1f}% / Spec '
             f'{fusion["specificity"]*100:.1f}% (cảnh báo = WARNING+)')
fig.colorbar(im, fraction=.046)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'confusion_4level_fusion.png'),
                                dpi=150); plt.close(fig)

# (3) Latency to alarm
ramps = ['ramp_+40/15ph', 'ramp_+40/30ph', 'ramp_+40/60ph',
         'ramp_+18/30ph_DUOI_NGUONG']
old = [summary['latency_to_alarm']['ramps'][r]['old_min'] or 0 for r in ramps]
new = [summary['latency_to_alarm']['ramps'][r]['new_min'] or 0 for r in ramps]
x = np.arange(len(ramps))
fig, ax = plt.subplots(figsize=(8.4, 3.8))
ax.bar(x - .2, old, .38, label='Hệ cũ (chỉ ngưỡng tĩnh)', color='#aaaaaa')
ax.bar(x + .2, new, .38, label='Golden Watch (EWMA+CUSUM)', color='#e8a020')
for i in range(len(ramps)):
    note = ('0/20 KHÔNG BAO GIỜ' if old[i] == 0 else f'{old[i]:.1f} ph')
    ax.text(i - .2, old[i] + .4, note, ha='center', fontsize=7.5)
    fired = summary['latency_to_alarm']['ramps'][ramps[i]]['new_fired']
    ax.text(i + .2, new[i] + .4, f'{new[i]:.1f} ph ({fired})',
            ha='center', fontsize=7.5)
ax.set_xticks(x, [r.replace('_', ' ') for r in ramps], fontsize=8)
ax.set_ylabel('Phút từ khi triệu chứng bắt đầu')
ax.set_title('Latency to Alarm — replay 20 đường dịch bệnh/gốc (seed 42)\n'
             'FAR 0/24h cả hai hệ; giảm 45–50% độ trễ phát hiện')
ax.legend(); ax.grid(axis='y', alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'latency_to_alarm.png'),
                                dpi=150); plt.close(fig)

# (4) MDR targets — FPR/FNR vs ngưỡng 5%/1%
mods = ['Face v3', 'Speech LOSO', 'Gait v2', 'Fusion (mô phỏng)']
fprs = [summary['face_v3']['FPR'], summary['speech_loso']['FPR'],
        summary['gait_v2']['FPR'], fusion['FPR']]
fnrs = [summary['face_v3']['FNR'], summary['speech_loso']['FNR'],
        summary['gait_v2']['FNR'], fusion['FNR']]
x = np.arange(len(mods))
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
for a, vals, tgt, name in zip(ax, [fprs, fnrs], [0.05, 0.01],
                              ['FPR — mục tiêu < 5%', 'FNR — mục tiêu < 1%']):
    bars = a.bar(mods, [v * 100 for v in vals],
                 color=['#2ca02c' if v <= tgt else '#d62728' for v in vals])
    a.axhline(tgt * 100, color='black', ls='--', lw=1.2,
              label=f'ngưỡng MDR {tgt*100:.0f}%')
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width()/2, v * 100 + .15, f'{v*100:.2f}%',
               ha='center', fontsize=8)
    a.set_title(name); a.set_ylabel('%'); a.legend()
    a.tick_params(axis='x', rotation=15); a.grid(axis='y', alpha=.3)
fig.suptitle('Kiểm chuẩn MDR (Sensitivity/Specificity hệ thống)', y=1.03)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'mdr_fpr_fnr.png'),
                                dpi=150, bbox_inches='tight'); plt.close(fig)

print('\nOUTPUT:', OUT)
print('FILES:', sorted(os.listdir(OUT)))
