# -*- coding: utf-8 -*-
"""
LEAK CHECK — kiểm chứng nghi ngờ leakage của block img//50 (SYS-28).

Giả thuyết: 50 frame liên tiếp ≈ 1.7s video → GƯƠNG MẶT CÙNG MỘT NGƯỜI
nằm ở CẢ train lẫn test (block kề nhau) → mô hình phi tuyến (HistGB)
"nhớ" hình xăm geometry cá nhân → AUC 1.000.

Phép thử: AUC OOF GroupKFold(5) TOÀN BỘ dữ liệu theo CỠ BLOCK TĂNG DẦN
(//50 → //200 → //500 → //1000). Nếu AUC giảm khi block thô hơn = leakage
đúng như nghi ngờ; số ở block thô nhất mới là ước lượng trung thực.

Chạy: PYTHONUTF8=1 python training/leak_check_blocksize.py
"""

import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import train_face_landmarker_v3 as v3

SEED = 42


def oof_auc(make_model, X, y, blocks):
    oof = np.zeros(len(y))
    for ti, vi in GroupKFold(5).split(X, y, blocks):
        sc = StandardScaler().fit(X[ti])
        m = make_model()
        m.fit(sc.transform(X[ti]), y[ti])
        if hasattr(m, 'predict_proba'):
            oof[vi] = m.predict_proba(sc.transform(X[vi]))[:, 1]
        else:
            oof[vi] = m.decision_function(sc.transform(X[vi]))
    return roc_auc_score(y, oof)


def main():
    rows = v3.scan_dataset()
    X, y, _b50, bs_names, paths = v3.extract_all(rows)
    nums = np.array([int(re.search(r'img_(\d+)', p).group(1)) for p in paths])

    asym_names, il, ir = v3.blend_asym_pairs(bs_names)
    n_bs = len(bs_names)
    L, R = 5 + np.array(il), 5 + np.array(ir)
    G4 = np.hstack([X[:, :5], X[:, L] - X[:, R],
                    X[:, n_bs + 5:n_bs + 8]])

    print()
    print(f'{"block":>8} | {"so block":>8} | {"LogReg C=0.1":>12} | '
          f'{"HistGB":>7}')
    print('-' * 48)
    out = []
    for G in (50, 200, 500, 1000):
        blocks = np.array([f'{y[i]}_{nums[i] // G}' for i in range(len(y))])
        a1 = oof_auc(lambda: LogisticRegression(C=0.1, max_iter=3000,
                                                random_state=SEED),
                     G4, y, blocks)
        a2 = oof_auc(lambda: HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
            random_state=SEED), G4, y, blocks)
        nblk = len(set(blocks))
        print(f'{f"//{G}":>8} | {nblk:>8} | {a1:>12.3f} | {a2:>7.3f}')
        out.append({'block': G, 'n_blocks': nblk, 'auc_logreg': round(a1, 4),
                    'auc_histgb': round(a2, 4)})
    print()
    print('KET LUAN: AUC giam khi block tho hon = leakage gan-trung')
    print('Khuyen nghi: bao cao theo block tho nhat (gan khoang cach "moi nguoi").')

    import json
    with open(v3.PROJECT + '/test_results/leak_check_blocksize.json', 'w',
              encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
