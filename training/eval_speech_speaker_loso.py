# -*- coding: utf-8 -*-
"""
S-1 (NK-03) — SPEECH EVAL THEO NGƯỜI THẬT: LOSO + CROSS-MIC
=============================================================
Vấn đề phát hiện 10/09/2026: train_speech_torgo.py chia train/test theo
`hash(speaker_dir)` mà speaker_dir là SESSION×MIC (wav_arrayMic_F03S01 ≠
wav_headMic_F03S01) → cùng 1 người có thể nằm ở cả train lẫn test.
Số AUC 0.992 là session-level. Script này đánh giá lại chuẩn người-level:

  P1 LOSO (headline): leave-one-SPEAKER-out trên người THẬT
     (FC01…M04; strip mic + hậu tố session). 2 model: LogisticRegression
     (đầu tuyến tính — bài học HistGB SYS-28) + MLP cùng kiến trúc production.
  P2 Per-mic breakdown: OOF chia theo arrayMic / headMic.
  P3 Cross-mic chéo: train chỉ arrayMic → test headMic (người ngoài fold),
     và chiều ngược lại — kiểm chứng "không phụ thuộc micro".

Protocol ghi TRƯỚC: subsample tối đa --max-per-session file/session (seed 42),
model đóng băng hyper-params, KHÔNG tune trên test.
Output: test_results/speech_speaker_loso_<ts>/ (CSV OOF + JSON + 3 PNG).
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

DEFAULT_TORGO = os.path.join(
    r'C:\Users\Admin\Documents\NCKHKT_26\fga_project\data\datasets\speech',
    'TORGO Dataset for Dysarthric Speech - Audio Files')

SEED = 42
SPEAKER_RE = re.compile(r'^([FM]C?\d+)')   # FC01S01→FC01 · F01→F01 · M04S02→M04


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (round(100 * (c - h) / d, 1), round(100 * p, 1),
            round(100 * (c + h) / d, 1))


def scan_sessions(torgo_path):
    """→ list dict: path(wav), class_dir, label, mic, session, speaker."""
    from train_speech_torgo import extract_features_from_audio  # noqa: F401
    rows = []
    for class_dir, label in (('F_Con', 0), ('M_Con', 0),
                             ('F_Dys', 1), ('M_Dys', 1)):
        base = os.path.join(torgo_path, class_dir)
        if not os.path.isdir(base):
            continue
        for session in sorted(os.listdir(base)):
            sdir = os.path.join(base, session)
            if not os.path.isdir(sdir):
                continue
            mic = 'arrayMic' if 'arrayMic' in session else 'headMic'
            core = re.sub(r'^wav_(arrayMic|headMic)_', '', session)
            m = SPEAKER_RE.match(core)
            if not m:
                continue
            for wav in sorted(glob.glob(os.path.join(sdir, '*.wav'))):
                rows.append({'path': wav, 'label': label, 'mic': mic,
                             'session': session, 'speaker': m.group(1)})
    return rows


def subsample(rows, max_per_session, rng):
    by = {}
    for r in rows:
        by.setdefault((r['session'], r['mic']), []).append(r)
    out = []
    for key, lst in sorted(by.items()):
        idx = rng.permutation(len(lst))[:max_per_session]
        out.extend(lst[i] for i in sorted(idx))
    return out


def extract_all(rows, log_every=100):
    from train_speech_torgo import extract_features_from_audio
    X, meta, fail = [], [], 0
    t0 = time.time()
    for i, r in enumerate(rows):
        v = extract_features_from_audio(r['path'])
        if v is None or (np.allclose(v, 0) and i % 7 == 0 and fail_check(v)):
            fail += 1
            continue
        X.append(v)
        meta.append(r)
        if (i + 1) % log_every == 0:
            print(f'  [{i+1}/{len(rows)}] {time.time()-t0:.0f}s', flush=True)
    X = np.array(X, dtype=np.float64)
    # loại các vector toàn 0 (extract lỗi ghi 0 — trung thực)
    keep = ~np.all(X == 0, axis=1)
    fail += int((~keep).sum())
    X, meta = X[keep], [m for m, k in zip(meta, keep) if k]
    print(f'Extract xong: {len(meta)} file dùng được, {fail} lỗi/bỏ')
    return X, meta


def fail_check(v):
    return len(v) == 48


# ----------------------------------------------------------------------
def train_logreg(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0,
                             random_state=SEED)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def train_mlp(Xtr, ytr, Xte, epochs=60):
    import torch
    from sklearn.preprocessing import StandardScaler
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    sc = StandardScaler().fit(Xtr)
    Xtr_t = torch.FloatTensor(sc.transform(Xtr))
    ytr_t = torch.LongTensor(ytr)
    Xte_t = torch.FloatTensor(sc.transform(Xte))
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], 256), torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(256, 128), torch.nn.BatchNorm1d(128),
        torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(128, 64), torch.nn.BatchNorm1d(64),
        torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(64, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    g = torch.Generator().manual_seed(SEED)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True,
                                     generator=g)
    net.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(Xte_t), dim=1)[:, 1].numpy()


def pooled_metrics(y, p):
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    auc = float(roc_auc_score(y, p)) if len(set(y)) == 2 else None
    # Youden trên OOF pooled
    thr_best, j_best = 0.5, -1
    for thr in np.unique(np.round(p, 3)):
        pred = p >= thr
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        if sens + spec - 1 > j_best:
            j_best, thr_best = sens + spec - 1, float(thr)
    pred = p >= thr_best
    tp = int(((pred) & (y == 1)).sum()); fn = int(((~pred.astype(bool)) & (y == 1)).sum())
    tn = int(((~pred.astype(bool)) & (y == 0)).sum()); fp = int(((pred) & (y == 0)).sum())
    sens_lo, sens, sens_hi = wilson(tp, tp + fn)
    spec_lo, spec, spec_hi = wilson(tn, tn + fp)
    return {'auc': auc, 'threshold': round(thr_best, 3),
            'sens': sens, 'sens_ci': [sens_lo, sens_hi],
            'spec': spec, 'spec_ci': [spec_lo, spec_hi],
            'n': len(y), 'pos': int(y.sum())}


def loso(X, meta, trainer, train_mask_mic=None):
    """OOF prob theo fold = NGƯỜI. train_mask_mic: lọc mic tập train."""
    speakers = sorted({m['speaker'] for m in meta})
    oof = np.full(len(meta), np.nan)
    folds = []
    for sp in speakers:
        te = np.array([m['speaker'] == sp for m in meta])
        tr = ~te.copy()
        if train_mask_mic is not None:
            tr &= train_mask_mic
        ytr = np.array([meta[i]['label'] for i in np.where(tr)[0]])
        if len(set(ytr.tolist())) < 2:
            oof[te] = 0.5
            folds.append({'speaker': sp, 'note': 'fold train 1 lớp — bỏ'})
            continue
        Xtr, ytr = X[tr], ytr
        Xte = X[te]
        p = trainer(Xtr, ytr, Xte)
        oof[te] = p
        folds.append({'speaker': sp, 'n_test': int(te.sum()),
                      'label': int(meta[int(np.where(te)[0][0])]['label'])})
    return oof, folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--torgo', default=DEFAULT_TORGO)
    ap.add_argument('--max-per-session', type=int, default=20)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--fast', action='store_true',
                    help='smoke test: 3 file/session, 5 epochs MLP')
    args = ap.parse_args()
    if args.fast:
        args.max_per_session, args.epochs = 3, 5

    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    rng = np.random.default_rng(SEED)

    print('== S-1 SPEAKER-LEVEL LOSO + CROSS-MIC (NK-03) ==')
    rows = scan_sessions(args.torgo)
    spk = sorted({r['speaker'] for r in rows})
    print(f'Sessions: {len(set((r["session"], r["mic"]) for r in rows))} · '
          f'File: {len(rows)} · Người thật: {len(spk)} → {spk}')
    rows = subsample(rows, args.max_per_session, rng)
    print(f'Subsampling ≤{args.max_per_session}/session → {len(rows)} file')
    X, meta = extract_all(rows)
    y = np.array([m['label'] for m in meta])
    mic = np.array([m['mic'] == 'arrayMic' for m in meta])
    print(f'Người sau subsample: {len(set(m["speaker"] for m in meta))} · '
          f'arrayMic {int(mic.sum())} · headMic {int((~mic).sum())}')

    out_dir = os.path.join(ROOT, 'test_results',
                           f'speech_speaker_loso_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)

    # ---------- P1: LOSO 2 model ----------
    print('\n[P1] LOSO người thật — LogReg …', flush=True)
    oof_lr, folds_lr = loso(X, meta, train_logreg)
    print('[P1] LOSO người thật — MLP (production arch) …', flush=True)
    oof_mlp, _ = loso(X, meta,
                      lambda a, b, c: train_mlp(a, b, c, epochs=args.epochs))

    res = {'protocol': 'LOSO true-speaker (S-1, NK-03)', 'seed': SEED,
           'max_per_session': args.max_per_session,
           'n_files': len(meta), 'n_speakers': len(set(m['speaker'] for m in meta)),
           'epochs_mlp': args.epochs, 'p1': {}, 'p2': {}, 'p3': {},
           'folds_lr': folds_lr}

    m_lr = pooled_metrics(y, oof_lr)
    m_mlp = pooled_metrics(y, oof_mlp)
    res['p1'] = {'logreg': m_lr, 'mlp': m_mlp}
    print(f"  LogReg : AUC {m_lr['auc']} thr {m_lr['threshold']} · "
          f"Sens {m_lr['sens']} {m_lr['sens_ci']} · Spec {m_lr['spec']} {m_lr['spec_ci']}")
    print(f"  MLP    : AUC {m_mlp['auc']} thr {m_mlp['threshold']} · "
          f"Sens {m_mlp['sens']} {m_mlp['sens_ci']} · Spec {m_mlp['spec']} {m_mlp['spec_ci']}")

    # ---------- P2: per-mic ----------
    res['p2'] = {'arrayMic_logreg': pooled_metrics(y[mic], oof_lr[mic]),
                 'headMic_logreg': pooled_metrics(y[~mic], oof_lr[~mic])}
    print(f"  OOF arrayMic AUC {res['p2']['arrayMic_logreg']['auc']} · "
          f"headMic AUC {res['p2']['headMic_logreg']['auc']}")

    # ---------- P3: cross-mic chéo ----------
    print('[P3] Cross-mic: train array→test head & head→array (LogReg+MLP)')
    for name, tmic in (('train_array_test_head', mic),
                       ('train_head_test_array', ~mic)):
        oof_c, _ = loso(X, meta, train_logreg, train_mask_mic=tmic)
        sel = ~tmic
        res['p3'][name + '_logreg'] = pooled_metrics(y[sel], oof_c[sel])
        oof_c2, _ = loso(X, meta,
                         lambda a, b, c: train_mlp(a, b, c, epochs=args.epochs),
                         train_mask_mic=tmic)
        res['p3'][name + '_mlp'] = pooled_metrics(y[sel], oof_c2[sel])
        print(f"  {name}: LogReg AUC {res['p3'][name + '_logreg']['auc']} · "
              f"MLP AUC {res['p3'][name + '_mlp']['auc']}")

    # ---------- CSV OOF ----------
    import csv
    with open(os.path.join(out_dir, 'oof_predictions.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['path', 'speaker', 'mic', 'label', 'p_logreg', 'p_mlp'])
        for i, m in enumerate(meta):
            w.writerow([os.path.basename(m['path']), m['speaker'], m['mic'],
                        m['label'], round(float(oof_lr[i]), 4),
                        round(float(oof_mlp[i]), 4)])

    with open(os.path.join(out_dir, 'summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    # ---------- FIGURES ----------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=170)
    for p, lbl, col in ((oof_lr, f"LogReg AUC={m_lr['auc']:.3f}", '#1a6faf'),
                        (oof_mlp, f"MLP AUC={m_mlp['auc']:.3f}", '#c0392b')):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2.2, color=col, label=lbl)
    ax.plot([0, 1], [0, 1], 'k:', lw=1)
    ax.set_xlabel('FPR'); ax.set_ylabel('Sensitivity')
    ax.set_title('Speech LOSO theo NGƯỜI THẬT (TORGO) — OOF pooled')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'roc_speaker_loso.png'))

    # per-speaker sens (dys) / spec (con)
    sps = sorted({m['speaker'] for m in meta})
    vals, names = [], []
    for sp in sps:
        sel = np.array([m['speaker'] == sp for m in meta])
        yy, pp = y[sel], oof_mlp[sel]
        if yy[0] == 1:
            thr = m_mlp['threshold']
            vals.append(100 * float(((pp >= thr).astype(int) == yy).mean()))
            names.append(sp + '\n(dys)')
        else:
            thr = m_mlp['threshold']
            vals.append(100 * float(((pp < thr).astype(int) == (1 - yy)).mean()))
            names.append(sp + '\n(con)')
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=170)
    ax.bar(range(len(vals)), vals,
           color=['#c0392b' if '(dys)' in n else '#1a6faf' for n in names])
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
    ax.axhline(50, color='gray', ls=':', lw=1)
    ax.set_ylim(0, 105); ax.set_ylabel('% đúng (sens dys / spec con)')
    ax.set_title(f"LOSO từng người — MLP @threshold {m_mlp['threshold']}")
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'per_speaker.png'))

    # mic comparison
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=170)
    labels = ['P2 array\n(OOF)', 'P2 head\n(OOF)', 'P3 array→head', 'P3 head→array']
    aucs = [res['p2']['arrayMic_logreg']['auc'],
            res['p2']['headMic_logreg']['auc'],
            res['p3']['train_array_test_head_logreg']['auc'],
            res['p3']['train_head_test_array_logreg']['auc']]
    ax.bar(labels, aucs, color=['#1a6faf', '#1a6faf', '#e67e22', '#e67e22'])
    for i, v in enumerate(aucs):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    ax.axhline(0.5, color='gray', ls=':', lw=1)
    ax.set_ylim(0, 1.05); ax.set_ylabel('AUC (LogReg)')
    ax.set_title('Microphone robustness — array vs head mic')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'mic_comparison.png'))

    print(f'\nĐã lưu: {out_dir} (CSV + JSON + 3 PNG)')
    return res


if __name__ == '__main__':
    main()
