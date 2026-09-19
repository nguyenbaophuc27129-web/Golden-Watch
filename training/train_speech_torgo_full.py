# -*- coding: utf-8 -*-
"""
SPEECH FULL-DATA FINAL MODEL (NK-12) — train 100% TORGO
=======================================================
Mục tiêu: khai thác 100% dataset speech (17,635 file TORGO) cho mô hình
cuối cùng, thay vì 1,100 file subsample của protocol LOSO.

Nguyên tắc trung thực:
  - Số công bố VẦN là LOSO người thật 0.62 (NK-03) + openSMILE 0.663
    (NK-06). Mô hình full-data KHÔNG có số "độ chính xác" mới (fit 100%).
  - Production module2 KHÔNG tự đổi: module2_main hard-code
    `speech_torgo_20260828_211130.pth` (ngưỡng 30/56% đã chốt từ
    extended test M2-10/M2-12). Artifact full là MÔ HÌNH NGHIÊN CỨU —
    muốn thay production phải chạy lại extended test threshold.
Resumable: trích đặc trưng lưu cache npz theo lô — chạy lại không mất.
Chạy: PYTHONUTF8=1 python training/train_speech_torgo_full.py [--fast]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (TRAIN_DIR, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_speech_speaker_loso import scan_sessions  # noqa: E402

CACHE = os.path.join(ROOT, 'test_results', '_speech_features_full.npz')
SEED = 42
EPOCHS = 150          # giống production train_speech_torgo.py
HIDDEN_DIMS = [256, 128, 64]


def extract_resumable(rows, batch=500, log=print):
    """Trích 48-ft cho TẤT CẢ file; cache npz bỏ qua file đã trích."""
    from train_speech_torgo import extract_features_from_audio

    done, feats, labels, paths = {}, [], [], []
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        for i, p in enumerate(d['paths']):
            done[p] = i
        feats, labels = list(d['X']), list(d['y'])
        paths = list(d['paths'])
        log(f'Cache: {len(paths)} file đã trích từ lần trước')
    todo = [r for r in rows if r['path'] not in done]
    log(f'Còn {len(todo)}/{len(rows)} file cần trích…')
    t0, n_fail = time.time(), 0
    for k, r in enumerate(todo):
        v = extract_features_from_audio(r['path'])
        if v is None or np.allclose(np.asarray(v, dtype=float), 0):
            n_fail += 1
            continue
        feats.append(np.asarray(v, dtype=np.float64))
        labels.append(r['label'])
        paths.append(r['path'])
        if (k + 1) % batch == 0:
            np.savez_compressed(CACHE, X=np.array(feats),
                                y=np.array(labels),
                                paths=np.array(paths, dtype=object))
            rate = (k + 1) / max(time.time() - t0, 1)
            eta = (len(todo) - k - 1) / max(rate, 1e-9) / 3600
            log(f'  [{k+1}/{len(todo)}] {rate:.1f} file/s · ETA {eta:.1f}h · '
                f'lỗi {n_fail}')
    np.savez_compressed(CACHE, X=np.array(feats), y=np.array(labels),
                        paths=np.array(paths, dtype=object))
    log(f'Trích xong: {len(paths)} file ({n_fail} lỗi/bỏ) · '
        f'{time.time()-t0:.0f}s')
    return np.array(feats), np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--torgo', default=os.path.join(
        r'C:\Users\Admin\Documents\NCKHKT_26\fga_project\data\datasets\speech',
        'TORGO Dataset for Dysarthric Speech - Audio Files'))
    ap.add_argument('--fast', action='store_true',
                    help='smoke: train trên cache hiện có, 5 epochs')
    args = ap.parse_args()
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    import torch
    from train_speech_torgo import DysarthriaClassifier
    from sklearn.preprocessing import StandardScaler

    print('== SPEECH FULL-DATA FINAL MODEL (NK-12) ==')
    rows = scan_sessions(args.torgo)
    print(f'TORGO: {len(rows)} file toàn bộ dataset')
    X, y = extract_resumable(rows)
    print(f'Train trên 100% data dùng được: {X.shape}')

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    sc = StandardScaler().fit(X)
    Xtr = torch.FloatTensor(sc.transform(X))
    ytr = torch.LongTensor(y)
    model = DysarthriaClassifier(input_dim=X.shape[1],
                                 hidden_dims=HIDDEN_DIMS)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(Xtr, ytr)
    g = torch.Generator().manual_seed(SEED)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True,
                                     generator=g)
    epochs = 5 if args.fast else EPOCHS
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f'  epoch {ep+1}/{epochs} loss {tot/len(dl):.4f}')

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    pth_out = os.path.join(ROOT, 'models', f'speech_torgo_full_{ts}.pth')
    torch.save(model.state_dict(), pth_out)
    meta = {
        'version': 'speech_torgo_full',
        'created': ts,
        'note': ('MÔ HÌNH NGHIÊN CỨU fit 100% TORGO (NK-12). Production '
                 'GIỮ speech_torgo_20260828_211130.pth (ngưỡng 30/56% đã '
                 'chốt). Số công bố: LOSO người thật NK-03.'),
        'n_files': int(len(y)),
        'n_dys': int((y == 1).sum()), 'n_con': int((y == 0).sum()),
        'hidden_dims': HIDDEN_DIMS, 'epochs': epochs, 'seed': SEED,
        'scaler_mean': sc.mean_.tolist(), 'scaler_scale': sc.scale_.tolist(),
        'ref_loso_auc': 0.620, 'ref_opensmile_auc': 0.663,
    }
    with open(os.path.join(ROOT, 'models', f'speech_torgo_full_{ts}.json'),
              'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f'Đã lưu: {pth_out} (+ JSON metadata)')


if __name__ == '__main__':
    main()
