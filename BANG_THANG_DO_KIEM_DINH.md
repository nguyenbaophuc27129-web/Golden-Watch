# BẢNG THANG ĐO KIỂM ĐỊNH MÔ HÌNH — CHUẨN AI + CHUẨN Y TẾ (CÓ TRÍCH BÀI BÁO)

> Tài liệu đối chiếu của Golden-Watch (PSCS). Mọi số liệu bên dưới là **số thật**
> từ `test_results/metrics_pack_20260907_223217/` (chạy 07/09/2026, script
> `training/evaluate_all_metrics.py`, seed 42, đánh giá out-of-fold/LOSO —
> không có số "điều trị trên tập train").

---

## 1. THANG ĐO PHÂN LOẠI (Classification Metrics)

| Thang đo | Công thức | Khi nào BẮT BUỘC dùng | Nguồn trích dẫn | Số liệu thật hiện có |
|---|---|---|---|---|
| **Accuracy** | (TP+TN)/N | Tổng quan; chỉ có ý nghĩa khi dữ liệu cân bằng | Chuẩn; CI theo Wilson 1927 | face_rules 57.1% [52.6–61.5] · face-ML 81.2% [77.4–84.4] · gait 88.9% [83.1–92.9] · speech 90.9% [80.4–96.1] |
| **Precision (PPV)** | TP/(TP+FP) | Khi báo động giả (FP) tốn kém — đúng bài toán alarm y tế | Chuẩn PPV trong chẩn đoán; STARD 2015 | face_rules 40.6% · face-ML 70.8% · gait 48.5% (ngưỡng Youden chấp nhận FP cao để không bỏ sót) · speech 86.7% |
| **Recall / Sensitivity** | TP/(TP+FN) | Khi bỏ sót bệnh nhân (FN) là chết người — **ưu tiên số 1 của PSCS** | Youden 1950; STARD 2015 | face_rules 65.6% · face-ML 72.6% · gait 94.1% [73.0–99.0] · speech 96.3% [81.7–99.3] |
| **Specificity** | TN/(TN+FP) | Khả năng "thanh nhiên" đúng | STARD 2015 | face_rules 53.0% · face-ML 85.4% · gait 88.3% · speech 85.7% |
| **NPV** | TN/(TN+FN) | Trấn an gia đình "không có chuyện gì" | STARD 2015 | face-ML 86.4% · gait 99.2% · speech 96.0% |
| **F1-Score** | 2PR/(P+R) | Dữ liệu lệch lớp | van Rijsbergen 1979 (Information Retrieval, ch.7) | face_rules 0.501 · face-ML 0.717 · gait 0.640 · speech 0.912 |
| **AUC-ROC** | diện tích dưới ROC | So sánh mọi ngưỡng, không phụ thuộc threshold | Hanley & McNeil 1982, *Radiology* 143:29-36; CI: DeLong 1988 / bootstrap | face_rules 0.638 [0.583–0.687] · **face-ML Logistic 0.845 [0.802–0.885]** · MLP 0.842 · gait LOSO 0.879 [0.742–0.961] · speech 0.992 [0.968–1.0] |
| **Wilson 95% CI** mọi tỷ lệ | Wilson score interval | BẮT BUỘC khi n nhỏ (gait 15 subject) | **Wilson EB. J Am Stat Assoc. 1927;22(158):209-212** | Áp dụng cho 100% các tỷ lệ trên |
| **Youden's J** | Sens + Spec − 1 | Chọn ngưỡng vận hành tối ưu | **Youden WJ. Cancer. 1950;3(1):32-35** | gait: ngưỡng 0.10, J=0.824 · face-ML J=0.580 |

**Điểm nhấn trung thực (hỏi hội đồng):** face-ML tốt hơn face-rules 0.638→0.845
AUC vì rules là ngưỡng tuyến tính cứng, còn mô hình học tổ hợp 5 đặc trưng.
Song **model chưa nối vào app** (M1-07: chờ protocol B — SYS-15) — không được
gọi số 0.845 là "số hệ thống đang chạy".

## 2. THANG ĐO HỒI QUY (Regression Metrics)

| Thang đo | Công thức | Dùng cho PSCS khi nào | Nguồn trích dẫn | Trạng thái số liệu |
|---|---|---|---|---|
| **MAE** | mean(\|y−ŷ\|) | Sai lệch NIHSS ước tính vs bác sĩ (đơn vị điểm NIHSS) | Willmott & Matsuura 2005, *Climate Research* 30:79-82 (MAE ổn định hơn RMSE với outlier) | **THIẾU — trung thực:** chưa có NIHSS gold-standard. Sẽ tính trên 50 video NIHSS chuẩn (L-01A) |
| **MSE / RMSE** | mean((y−ŷ)²), √MSE | Phạt nặng sai lệch lớn (bỏ sót nặng) | Chuẩn hồi quy | THIẾU (như trên) |
| **R²** | 1 − SSres/SStot | Mô hình giải thích bao nhiêu % biến động | Chuẩn hồi quy | THIẾU (như above) |
| **Bland–Altman agreement** | bias = mean(y−ŷ); LoA = bias ± 1.96·SD | **Chuẩn y tế vàng** khi so 2 cách đo (ước tính máy vs bác sĩ) | **Bland & Altman. Lancet. 1986;1(8476):307-310** | THIẾU — chờ L-01A; protocol đã ghi QUY_TRINH_KIEM_DINH_Y_KHOA.md |
| **Weighted kappa** | κ có trọng số bậc | Trùng khớp thứ bậc điểm NIHSS (ordinal!) | **Cohen J. Psychol Bull. 1968;70(4):213-220** | THIẾU — chờ L-01A |

> **Vì sao không bịa số hồi quy:** không tồn tại dataset công khai có NIHSS
> do bác sĩ chấm từng frame khớp với ảnh của ta. Đưa số MAE/RMSE/R² không có
> nguồn so sánh là **gian lận khoa học** — thay vào đó công bố rõ "đang chờ
> kiểm định chéo với anchor ngoài".

## 3. THANG ĐO PHÂN CỤM (Clustering Metrics)

| Thang đo | Ý nghĩa | Nguồn trích dẫn | Số liệu thật |
|---|---|---|---|
| **Silhouette Score** (−1→1) | Cụm nội khớp chặt, liên cụm xa | **Rousseeuw PJ. J Comput Appl Math. 1987;20:53-65** | Gait 8 đặc trưng (chuẩn hóa): **2 lớp control/Parkinson = 0.681** (tách tốt); 3 nhóm trẻ/già/Parkinson = 0.172 (trẻ↔già khỏe chồng nhau — đúng y tế: cả 2 đều bình thường, chỉ Parkinson khác biệt). Face 5 đặc trưng 2 lớp = 0.049 → **2 lớp chồng nhau nhiều** → giải thích vì sao CẦN mô hình học (logistic) thay vì dùng ngưỡng/k-means |

Biểu đồ kèm: `gait_silhouette_pca.png` (PCA 2D 3 nhóm màu).

## 4. THANG ĐO BỔ TRỢ CHUẨN Y TẾ (đã/kế hoạch áp dụng)

| Công cụ | Vai trò | Trích dẫn | Trạng thái |
|---|---|---|---|
| **Cohen's d ± CI** (effect size) | Dataset face CÓ tín hiệu y khoa không (trước khi train) | **Cohen J. Statistical Power Analysis, 2nd ed. 1988**; SE: Hedges & Olkin 1985 | ✅ mouth_ratio d=0.759 (M1-06); forest plot `face_forest_effects.png` |
| **Calibration curve** | Xác suất dự đoán có "tin được" không | Hosmer–Lemeshow 1980 (*Commun Stat A9:1043-1069*); Steyerberg, *Clinical Prediction Models* 2009 | ✅ `face_ml_calibration.png` |
| **McNemar test** | So 2 mô hình trên CÙNG test set (rules vs ML) | **McNemar Q. Psychometrika. 1947;12(2):153-157** | Kế hoạch: chạy lại 2 model trên cùng block-split rồi test |
| **LOSO / GroupKFold theo block** | Chống leakage (đồng nhất mức subject/ảnh liên tiếp) | Varoquaux 2018 ("Cross-validation failure", *NeuroImage* 178:681-692) | ✅ gait LOSO 15 fold; face GroupKFold(5) block |
| **Class-weight / pos_weight** | Dữ liệu lệch lớp | Chuẩn (King & Zeng 2001, *Political Analysis* 9:137-163 — rare-event correction) | ✅ mọi model |

## 5. TIÊU CHUẨN BÁO CÁO KHOA HỌC ÁP DỤNG

| Chuẩn | Dùng để | Trích dẫn |
|---|---|---|
| **TRIPOD+AI** | Báo cáo mô hình dự đoán lâm sàng minh bạch | Collins GS et al. *BMJ* 2024;385:e078378 (TRIPOD gốc: BMJ 2015;350:g7594) |
| **STARD 2015** | Báo cáo độ chính xác chẩn đoán | Cohen JF et al. *BMJ* 2016;352:h5527 |
| **Saver 2006** | Cơ sở "thời gian vàng" | Saver JL. *Stroke* 2006;37(1):263-266 |

## 6. BẢNG TỔNG — MODULE × THANG ĐO (đủ/thiếu)

| Module | Classification đủ 7 chỉ số + CI | Phân cụm | Hồi quy/đồng thuận | Khoảng trống |
|---|---|---|---|---|
| **Face (rules, đang chạy app)** | ✅ n=478 | ✅ 0.049 | — | NO_FACE 38% do dataset (L-01) |
| **Face (ML 5-feat M1-07)** | ✅ n=478, AUC 0.845 | ✅ | — | Chưa nối app — chờ protocol B |
| **Gait v2 (LOSO PhysioNet)** | ✅ window + subject | ✅ 0.681 / 0.172 | — | Parkinson chỉ là proxy; ngưỡng M4-06 |
| **Speech (TORGO)** | ✅ n=55, AUC 0.992 | — | — | Tiếng Anh; thiếu tiếng Việt (L-02) |
| **Arm** | ❌ model synthetic, chưa có eval thật | — | — | **A-01: cần video thật** |
| **NIHSS estimator** | — | — | ❌ THIẾU (chờ 50 video L-01A) | Bland–Altman + wκ + MAE |
| **Radar** | ❌ chờ test hardware 08/09 | — | — | M5-01 |

## 7. NGUYÊN TẮC CHỐNG TỰ CHỨNG MINH (SYS-15)

1. **Mọi số kèm CI 95% + n + ngưỡng + phương pháp đánh giá** (out-of-fold/LOSO/same-data).
2. Không công bố same-data AUC như kết quả hệ thống (face same-data 0.856≠0.845 OOF — ghi cả hai nhưng nhãn rõ).
3. Số không có nguồn data thật → ghi **THIẾU**, không nội suy.
4. Mỗi thí nghiệm lưu script + JSON + CSV + PNG (tái lập được, seed 42).

---
*Bộ số liệu: `test_results/metrics_pack_20260907_223217/` — 6 dòng kết quả,
13 biểu đồ PNG, seed 42, chạy 07/09/2026.*
