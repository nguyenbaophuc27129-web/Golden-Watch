# -*- coding: utf-8 -*-
"""
TRAIN FACE LANDMARKER v3.1 — CẢI THIỆN HỢP LỆ TỐI ĐA trên nền v3 (SYS-28).

Mục tiêu user: "cải thiện model tốt nhất, train dataset vào, chỉnh thuật toán
nhận diện để test chính xác". LƯU Ý TRUNG THỰC: KHÔNG AI đạt 100% test trên
dữ liệu y khoa thật — 100% = dấu hiệu leakage/overfit (TRIPOD+AI sẽ loại).
v3.1 therefore:
  1. THÊM THUẬT TOÁN: LogisticReg / SVM-RBF / RandomForest / HistGradientBoost
     / MLP nhỏ — chọn cấu hình TỐT NHẤT bằng OOF GroupKFold(5) trên TRAIN
     (test vẫn chỉ chạm 1 lần, đúng protocol v3).
  2. CLASS WEIGHT 'balanced' — dataset lệch (1241/2499) làm sens bị kéo xuống.
  3. HAI NGƯỠNG NGHIỆM: Youden J (cân bằng) + NGƯỠNG SÀNG LỌC sens>=90% OOF
     (ưu tiên y khoa: không bỏ sót người bệnh — chấp nhận giảm spec).
  4. AUC 95% CI bootstrap trên test + OOF 5-fold toàn bộ dữ liệu cho cấu hình
     cuối (2 ước lượng độc lập để đối chiếu).

Tái dùng pipeline trích xuất của v3 (import — cùng đặc trưng, cùng seed 42).
Chạy:  PYTHONUTF8=1 python training/train_face_landmarker_v31.py
"""

import os
import sys
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_face_landmarker_v3 as v3

SEED = 42
PROJECT = v3.PROJECT
rng = np.random.RandomState(SEED)


# ======================================================================
# CẦU HÌNH THUẬT TOÁN (mỗi mục = (tên, hàm tạo model, có class_weight?))
# ======================================================================
def model_zoo():
    zoo = []
    for C in (0.03, 0.1, 0.3, 1.0, 3.0):
        zoo.append((f'LogReg_C{C}', lambda C=C: LogisticRegression(
            C=C, max_iter=3000, random_state=SEED), False))
        zoo.append((f'LogReg_C{C}_bal', lambda C=C: LogisticRegression(
            C=C, max_iter=3000, random_state=SEED, class_weight='balanced'),
            False))
    for C in (0.5, 1.0, 3.0):
        zoo.append((f'SVM_rbf_C{C}', lambda C=C: SVC(
            C=C, kernel='rbf', random_state=SEED), True))
    zoo.append(('RandomForest_400', lambda: RandomForestClassifier(
        n_estimators=400, min_samples_leaf=3, random_state=SEED,
        n_jobs=-1), True))
    zoo.append(('HistGB_200', lambda: HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
        random_state=SEED), False))
    zoo.append(('MLP_32_16', lambda: MLPClassifier(
        hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=800,
        early_stopping=True, random_state=SEED), True))
    return zoo


def scores_of(model, X):
    """Điểm liên tục cho AUC/ngưỡng: decision_function nếu có, else proba."""
    if hasattr(model, 'decision_function'):
        s = model.decision_function(X)
        return (s - s.min()) / (np.ptp(s) + 1e-12)  # scale [0,1] cùng thang
    return model.predict_proba(X)[:, 1]


def oof_scores(make_model, X, y, blocks):
    oof = np.zeros(len(y))
    for ti, vi in GroupKFold(5).split(X, y, blocks):
        sc = StandardScaler().fit(X[ti])
        m = make_model()
        m.fit(sc.transform(X[ti]), y[ti])
        oof[vi] = scores_of(m, sc.transform(X[vi]))
    return oof


def threshold_for_sens(y, s, target=0.90):
    """Ngưỡng THẤP NHẤT để OOF sensitivity >= target (điểm sàng lọc)."""
    order = np.argsort(-s)
    ys = y[order]
    tp = np.cumsum(ys)
    sens = tp / max(ys.sum(), 1)
    hit = np.where(sens >= target)[0]
    k = hit[0] if len(hit) else len(s) - 1
    return float(s[order][k])


def metrics_at(y, s, thr):
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {'thr': round(float(thr), 4), 'sens': round(sens, 4),
            'spec': round(spec, 4), 'f1': round(f1, 4),
            'cm': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}}


def auc_ci_bootstrap(y, s, n=1000):
    """CI 95% percentile bootstrap cho AUC test (seed cố định)."""
    br = np.random.RandomState(SEED)
    n_pos = int((y == 1).sum())
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    aucs = []
    for _ in range(n):
        p = br.choice(idx_pos, n_pos, replace=True)
        n_ = br.choice(idx_neg, len(idx_neg), replace=True)
        aucs.append(roc_auc_score(y[list(p) + list(n_)], s[list(p) + list(n_)]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    print('=' * 66)
    print('TRAIN FACE LANDMARKER v3.1 — mo rong thuat toan + nguong sang loc')
    print('=' * 66)
    rows = v3.scan_dataset()
    stamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(PROJECT, 'test_results',
                           f'face_landmarker_v31_{stamp}')
    os.makedirs(run_dir, exist_ok=True)

    X, y, blocks, bs_names, _paths = v3.extract_all(rows)
    asym_names, il, ir = v3.blend_asym_pairs(bs_names)
    n_bs = len(bs_names)
    L = 5 + np.array(il)
    R = 5 + np.array(ir)
    asym_X = X[:, L] - X[:, R]
    pose_X = X[:, n_bs + 5:n_bs + 8]
    np.savez_compressed(os.path.join(run_dir, 'features.npz'),
                        X=X, y=y, blocks=blocks)
    groups = {
        'G1_5ratios': (X[:, :5], v3.RATIO_NAMES),
        'G3_blend_asym_pose': (np.hstack([asym_X, pose_X]),
                               asym_names + ['yaw', 'pitch', 'roll']),
        'G4_all': (np.hstack([X[:, :5], asym_X, pose_X]),
                   v3.RATIO_NAMES + asym_names + ['yaw', 'pitch', 'roll']),
    }

    tr_groups = np.unique(blocks)
    gs = rng.permutation(tr_groups)
    tr_g = set(gs[:int(len(gs) * 0.8)])
    tr_m = np.array([b in tr_g for b in blocks])
    te_m = ~tr_m
    tr_idx = np.where(tr_m)[0]

    # --- CHỌN (nhóm × thuật toán) TỐT NHẤT bằng OOF trên TRAIN ----------
    table, best = [], None
    zoo = model_zoo()
    for tag, (Xg, names) in groups.items():
        for mname, make, has_cw in zoo:
            s_oof = oof_scores(make, Xg[tr_idx], y[tr_idx], blocks[tr_idx])
            auc = roc_auc_score(y[tr_idx], s_oof)
            table.append({'group': tag, 'model': mname,
                          'auc_oof_train': round(auc, 4)})
            if best is None or auc > best['auc']:
                best = {'group': tag, 'model': mname, 'make': make,
                        'names': names, 'auc': auc, 's_oof': s_oof}
        print(f'  xong nhom {tag} ({len(table)} cau hinh)')

    # --- FIT CUỐI trên train + CHẠM TEST 1 LẦN ---------------------------
    b = best
    sc = StandardScaler().fit(groups[b['group']][0][tr_idx])
    model = b['make']()
    model.fit(sc.transform(groups[b['group']][0][tr_idx]), y[tr_idx])
    s_te = scores_of(model, sc.transform(groups[b['group']][0][te_m]))
    y_te = y[te_m]
    auc_te = roc_auc_score(y_te, s_te)
    lo, hi = auc_ci_bootstrap(y_te, s_te)

    fpr, tpr, thr = roc_curve(y[tr_idx], b['s_oof'])
    j = int(np.argmax(tpr - fpr))
    thr_youden = float(thr[j])
    thr_s90 = threshold_for_sens(y[tr_idx], b['s_oof'], 0.90)

    m_you = metrics_at(y_te, s_te, thr_youden)
    m_s90 = metrics_at(y_te, s_te, thr_s90)

    # Ước lượng thứ 2: OOF 5-fold GroupKFold TRÊN TOÀN BỘ dữ liệu
    s_all = oof_scores(b['make'], groups[b['group']][0], y, blocks)
    auc_all = roc_auc_score(y, s_all)

    print('=' * 66)
    print(f'TOT NHAT: {b["group"]} + {b["model"]} | OOF train AUC '
          f'{b["auc"]:.3f}')
    print(f'TEST (block-aware): AUC {auc_te:.3f} (95% CI {lo:.3f}-{hi:.3f})')
    print(f'  Youden J  : sens {m_you["sens"] * 100:.1f}% spec '
          f'{m_you["spec"] * 100:.1f}% F1 {m_you["f1"]:.3f}')
    print(f'  Sang loc sens>=90 (OOF): sens {m_s90["sens"] * 100:.1f}% spec '
          f'{m_s90["spec"] * 100:.1f}% (thr {m_s90["thr"]:.3f})')
    print(f'OOF 5-fold TOAN BO du lieu: AUC {auc_all:.3f}')

    results = {'seed': SEED, 'best': {'group': b['group'],
               'model': b['model']},
               'test_auc': round(auc_te, 4), 'test_auc_ci95':
                   [round(lo, 4), round(hi, 4)],
               'youden': m_you, 'screen_sens90': m_s90,
               'oof_all_data_auc': round(auc_all, 4),
               'selection_table': sorted(
                   table, key=lambda r: -r['auc_oof_train'])[:15]}
    with open(os.path.join(run_dir, 'results.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # ROC PNG
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"OOF train ({b['model']})")
    fpr_t, tpr_t, _ = roc_curve(y_te, s_te)
    plt.plot(fpr_t, tpr_t, lw=2,
             label=f'Test AUC {auc_te:.3f} [{lo:.3f}-{hi:.3f}]')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f"v3.1 {b['group']} + {b['model']}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'roc_v31.png'), dpi=150)
    plt.close()

    # Artifact JSON (metadata; model nhị phân lưu .pkl cạnh đó)
    art = {'version': 'face_blend_v31', 'created': stamp,
           'group': b['group'], 'model_type': b['model'],
           'features': b['names'], 'scaler_mean':
               [round(float(v), 6) for v in sc.mean_],
           'scaler_scale': [round(float(v), 6) for v in sc.scale_],
           'threshold_youden': round(thr_youden, 4),
           'threshold_screen_sens90': round(thr_s90, 4),
           'auc_oof_train': round(b['auc'], 4), 'auc_test':
               round(auc_te, 4), 'auc_test_ci95': [round(lo, 4),
                                                   round(hi, 4)],
           'youden': m_you, 'screen_sens90': m_s90,
           'auc_oof_all_data': round(auc_all, 4),
           'note': 'CHUA qua protocol B (SYS-15) — khong bat tren app'}
    art_path = os.path.join(PROJECT, 'models', f'face_blend_v31_{stamp}.json')
    with open(art_path, 'w', encoding='utf-8') as f:
        json.dump(art, f, ensure_ascii=False, indent=1)
    try:
        import joblib
        joblib.dump({'model': model, 'scaler': sc},
                    art_path.replace('.json', '.pkl'))
    except Exception as e:
        print(f'(bo qua luu pkl: {e})')

    print(f'Ket qua : {run_dir}')
    print(f'Artifact: {art_path}')
    print('Luu y trung thuc: KHONG the/KHONG nen dat 100% test — '
          '100% = Leakage. muc tieu y khoa: sens >= 90% khong bo sot.')


if __name__ == '__main__':
    main()
