# DÀN Ý BÁO CÁO DỰ THI (≤15 TRANG A4 — ĐÚNG KH 5.2.2)

> Định dạng: Times New Roman 14, dòng đơn, lề trái 3cm/phải 2cm/trên-dưới 2cm,
> **KHÔNG ghi tên đơn vị**. Mọi số dưới đây là số thật từ:
> `test_results/metrics_pack_20260907_223217/` + `prototype_benchmark_20260908_050756.json`.
> Chỗ [ĐIỀN] = cần bổ sung khi có dữ liệu/hardware.

## TRANG BÌA (trang 1)
- Lĩnh vực: **5 — Kỹ thuật Y Sinh** (Thiết bị Y Sinh)
- Tên: Golden Watch — Hệ giám sát đa cảm biến không đeo phát hiện sớm đột quỵ tại nhà và kiểm soát báo động giả
- Người thực hiện / người bảo trợ / người hướng dẫn / thời gian–địa điểm
  [ĐIỀN: tên đội, GVHD, thời gian nghiên cứu 01/2026→10/2026 — phải khớp sổ nhật ký]

## TÓM TẮT (trang 2) — 4 tính bắt buộc của KH
- **Tính mới:** 4-Layer Defense chống báo giả + fusion 5 tín hiệu (mặt, nói, tay, dáng đi, radar mmWave) + cầu nối NIHSS→bệnh viện (QR/PDF). Phát hiện khoa học: dataset công khai chứa tín hiệu méo mặt phân biệt đột quỵ ở mức đặc trưng (Cohen's d=0.759) nhưng KHÔNG ở tọa độ thô (AUC 0.555) — bài học biểu diễn đặc trưng.
- **Tính khoa học:** mọi số liệu đánh giá out-of-fold/LOSO kèm Wilson 95% CI, chuẩn TRIPOD+AI/STARD 2015 (BANG_THANG_DO_KIEM_DINH.md).
- **Tính thực tiễn:** chi phí phần cứng mua thêm ≈1,01 triệu đồng; chạy local, không cần internet, bảo vệ riêng tư (radar không camera).
- **Tính cộng đồng:** dành cho người cao tuổi sống một mình; hướng tới giảm tử vong/đ t quỵ bằng thời gian vàng <4,5h (Saver 2006: 1,9 triệu neuron/phút).

## 1. LÝ DO CHỌN DỰ ÁN (trang 3)
- Số liệu gánh nặng bệnh tật VN: đột quỵ hàng đầu về tử vong-tàn phốt [ĐIỀN: Bộ Y tế/Bộ Y Tế số mới nhất]
- Người cao tuổi sống một mình phát hiện muộn → phần lớn đến viện quá cửa sổ rt-PA
- Cơ sở khoa học FAST + NIHSS item 4/5/6/10 có thể đo từ camera/radar

## 2. VẤN ĐỀ NGHIÊN CỨU + GIẢ THUYẾT (trang 3-4)
- Vấn đề: Xây dựng hệ thống giám sát tại nhà có khả năng phát hiện dấu hiệu đột quỵ đa mô hình với tỷ lệ bỏ sót thấp và báo giả có kiểm soát?
- Giả thuyết H1: tổ hợp 5 tín hiệu + luật kết hợp (R1/R2) phát hiện tình huống mô phỏng tốt hơn từng tín hiệu riêng.
- Giả thuyết H2: chống báo giả bằng 4 tầng (baseline cá nhân, ngữ cảnh, luật FAST, ngưỡng kết hợp) giữ EMERGENCY giả ≤5%.
- Kết quả hiện có: H1 → fusion + defense [ĐIỀN sau protocol B]; H2 → thiết kế sẵn, kiểm chứng SYS-15b.

## 3. THIẾT KẾ VÀ PHƯƠNG PHÁP (trang 4-7)
- Kiến trúc PSCS: 5 module → 4-Layer Defense → Fusion (trọng số arm .30, face/speech .20, gait/radar .15) → Triage → NIHSS 4 items (tối đa 13) → Alert/Handoff QR-PDF
- Prototype: RTX 3050 (máy chủ) + C270 + LD2450 — bảng BOM, sơ đồ, lắp đặt (PROTOTYPE_SPEC.md)
- Phương pháp đánh giá: LOSO 15 subject (gait), GroupKFold-5 theo block (face), median-3 cửa sổ (speech), Wilson CI, bootstrap AUC CI — **đăng ký trước khi chạy** (pre-registration)
- **Rủi ro + an toàn** (bắt buộc KH): ngã thử trên nệm + người bảo hộ; dữ liệu xử lý tại máy; disclaimer không phản chẩn đoán; GỌI 115

## 4. TIẾN HÀNH NGHIÊN CỨU — SỐ LIỆU THẬT (trang 8-12) ⭐ mục 20đ
### 4.1 Kiểm định từng module (bảng 1 — từ metrics_pack)
| Module | Data | n | Sens [CI95] | Spec [CI95] | F1 | AUC [CI95] |
|---|---|---|---|---|---|---|
| Méo mặt (rules) | Kaggle face block-split | 478 | 65.6 [57.9–72.6] | 53.0 [47.5–58.4] | 0.50 | 0.638 [0.583–0.687] |
| Méo mặt (ML 5-feat) | cùng block-split | 478 | 72.6 [65.2–79.0] | 85.4 [81.1–88.8] | 0.72 | **0.845 [0.802–0.885]** |
| Dáng đi (LOSO) | PhysioNet 15 subject | 162 win | 94.1 [73.0–99.0] | 88.3 [82.0–92.6] | 0.64 | 0.879 [0.742–0.961] |
| Nói khó (TORGO) | 55 session | 55 | 96.3 [81.7–99.3] | 85.7 [68.5–94.3] | 0.91 | 0.992 [0.968–1.0] |
- Phân cụm: silhouette gait 2 lớp 0.681; face 0.049 → giải thích vì sao cần mô hình học
- **Bộ 5 biểu đồ định lượng** (`test_results/charts_accuracy_20260908_052303/`):
  A_ROC-gộp · B_Precision-Recall · C_bar-metrics±Wilson-CI · D_Sens-Spec-theo-ngưỡng (Youden) · E_forest-AUC±CI-bootstrap — chèn trực tiếp vào báo cáo/poster
- Hồi quy NIHSS (MAE/RMSE/R²/Bland–Altman/weighted-κ): [ĐIỀN sau SYS-18 — 50 video NIHSS chuẩn]
### 4.2 Hiệu năng prototype (bảng 2 — từ benchmark)
- Chu kỳ phân tích ≈ **81 ms** (face 15 + arm 28 + gait 17 + fusion ~0) so với chu kỳ app 3000 ms → dư địa 2919 ms
- Camera C270: 1280×720 @ 32 fps đọc liên tục; GPU RTX 3050 6GB [ĐIỀN VRAM dùng]
- Radar: baud 256000, chu kỳ 600 ms/nhịp
### 4.3 Chế tạo & kiểm tra hệ thống
- 58/58 unit test (`src/test_all_metrics.py`); 10/10 pre-flight (`test_sys14b_regression.py`)
- 10 kịch bản radar thật + 100 tình huống chống báo giả: [ĐIỀN sau M5-02 + SYS-15b]
- Kịch bản nghiệm thu 6 bước: KICH_BAN_NGHIEM_THU.md
### 4.4 Kết luận khoa học theo giả thuyết
- H1: [ĐIỀN]; H2: [ĐIỀN]. Trung thực: arm chưa có eval trên dữ liệu thật; speech chưa tiếng Việt (đang thu L-02)

## 5. KẾT LUẬN + HƯỚNG PHÁT TRIỂN (trang 13)
- Đã chứng minh được gì / chưa; giới hạn (proxy Parkinson, dataset công khai, không lâm sàng)
- Phát triển: thu tiếng Việt, cảm biến đeo, kết nối 115/bệnh viện tuyến, nghiên cứu lâm sàng có kiểm soát

## TÀI LIỆU THAM KHẢO (trang 14-15) — ≥5, đã có sẵn:
1. Saver JL. Time is brain—quantified. *Stroke* 2006;37(1):263-266
2. Hausdorff JM et al. Gait in Aging and Disease Database v1.0.0. *PhysioNet*
3. Collins GS et al. TRIPOD+AI. *BMJ* 2024;385:e078378
4. Cohen JF et al. STARD 2015. *BMJ* 2016;352:h5527
5. Wilson EB. Probable inference... *JASA* 1927;22(158):209-212
6. Bland JM, Altman DG. Lancet 1986;1(8476):307-310
7. Youden WJ. Cancer 1950;3(1):32-35 — [thêm báo cáo đột quỵ VN khi ĐIỀN mục 1]

## KẸP KÈM HỒ SƠ (KH 5.2.2)
- Sản phẩm: video <3 phút + ảnh prototype
- Phụ lục 1: khai báo AI tạo sinh | Phụ lục 2: sổ nhật ký (từ LOI_SO_MODULE.md) | Phụ lục 3: poster online
- Khai báo kế thừa fga_project (nếu có) — PR-03
