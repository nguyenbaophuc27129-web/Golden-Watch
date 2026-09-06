# QUY TRÌNH KIỂM ĐỊNH Y KHOA — PSCS GOLDEN-WATCH
> Tài liệu chuẩn bị cho SYS-07 — lập 06/09/2026
> **Trạng thái trung thực hiện tại: CHƯA có validation lâm sàng thật (0 thư bác sĩ).**
> Toàn bộ số liệu đang có là: TORGO thật (speech), dữ liệu nội bộ tự thu (face),
> dữ liệu mô phỏng (arm/gait/radar). Hồ sơ dưới đây chuẩn bị đầy đủ để thực hiện.

---

## 1. MỤC TIÊU KIỂM ĐỊNH (theo lịch Tuần 3-4)

| Chỉ số | Mục tiêu | Công cụ có sẵn |
|---|---|---|
| NIHSS ước tính vs bác sĩ | **r ≥ 0.85** (Pearson), MAE < 2 | `src/fusion/validation_metrics.py::evaluate_nihss_study` |
| Phát hiện đột quỵ | **Sens > 90%, Spec > 95%, F1** | `evaluate_detection_study` |
| So sánh 2 hệ thống | p < 0.05 | `paired_sign_test` |
| Chọn ngưỡng | Youden J tối đa | `find_youden` |

## 2. NGUYÊN TẮC ĐẠO ĐỨC

1. KHÔNG dùng trên người bệnh thật khi chưa có sự chấp thuận của bác sĩ.
2. Dữ liệu nghiên cứu ẩn danh (không tên, không địa chỉ trong dataset nghiên cứu).
3. Hệ thống luôn hiển thị disclaimer "KHÔNG PHẢN CHẨN ĐOÁN — GỌI 115".
4. Thư xác nhận bác sĩ là **consultation/review**, KHÔNG phải clinical trial
   (đúng hướng dẫn lịch: "KHÔNG phải clinical trial").

## 3. PHẦN A — VALIDATION NIHSS (50 video công khai)

**Nguồn video:** video chụp lâm sàng NIHSS công khai (YouTube giáo dục đại học
y — ví dụ chuỗi NIHSS training của các trung tâm đột quỵ; ghi nguồn từng video
vào `validation_data/video_sources.csv`).

**Quy trình 6 bước:**
1. Chọn 50 video NIHSS chuẩn (có điểm bác sĩ công bố nếu được; không thì mời bác sĩ chấm).
2. Chạy pipeline PSCS trên từng video: trích ảnh mặt (M1) + đoạn nói (M2) + cảnh tay (M3) → `nihss_est.csv` (dùng `estimate_nihss`).
3. Bác sĩ chấm NIHSS 4 items trên form in → `gold.csv` (10 case đầu mời bác sĩ đồng chấm 2 lần để đo inter-rater).
4. Chạy `evaluate_nihss_study(est, gold)` → xuất `validation_report_nihss.json` (r, MAE, within_1_point, verdict).
5. Vẽ scatter plot est vs gold (matplotlib) → `validation_fig_nihss_scatter.png`.
6. Nếu r < 0.85: phân tích từng item sai band, điều chỉnh `PROB_TO_ITEM`, chạy lại (không chỉnh trên tập test riêng của bác sĩ để tránh overfit).

## 4. PHẦN B — VALIDATION PHÁT HIỆN (100 scenarios)

| Nhóm | Số lượng | Kịch bản | Mục tiêu |
|---|---|---|---|
| A — False alarm | 50 | cười 10x, tập 10x, ngáp 10x, nói bình thường 10x, nhiễu 10x | FPR < 5% |
| B — True positive | 30 | mô phỏng dấu hiệu (theo hướng dẫn bác sĩ, KHÔNG gây nguy hiểm) | TPR > 90% |
| C — Video TORGO/dataset công khai | 20 | normal 10 + dysarthria 10 | đúng như kết quả 06/09 |

Chạy `evaluate_detection_study(y_true, fused_scores)` → Sens/Spec/F1 + ROC.

## 5. PHẦN C — THƯ XÁC NHẬN BÁC SĨ (2 thư mục tiêu)

- BV Nhân dân 115 — Trung tâm Đột quỵ (Trưởng khoa/BS Thần kinh)
- BV Đại học Y Dược TP.HCM (BS Thần kinh) — kèm 10 sample cases để feedback
- Phương án dự phòng: bác sĩ thần kinh tư/nghỉ hưu.
→ Mẫu thư: `MAU_THU_XAC_NHAN_BAC_SI.md` (in, mang video demo + báo cáo validation đi).

## 6. LỊCH TRÌNH

| Mốc | Việc | Trạng thái |
|---|---|---|
| Đã xong 06/09 | Tool validation + protocol + mẫu thư | ✅ |
| Tuần 3 (16-22/09) | Phần A: 50 video + xuất kết quả | 🔴 |
| Tuần 4 (23-29/09) | Gặp bác sĩ, 2 thư + feedback 10 cases | 🔴 |
| Tuần 4 | Phần B: 100 scenarios tại nhà tình nguyện | 🔴 |

## 7. KHẨN TRƯƠNG HỢP PHÁP (FAQ phỏng vấn)

**H: Em có chẩn đoán đột quỵ không?**
Đ: Không. Em xây dựng công cụ SÀNG LỌC HỖ TRỢ phát hiện sớm, luôn khuyến cáo
gọi 115; NIHSS là ước tính để ưu tiên chuyển tuyến, bác sĩ mới là người chấm
chuẩn. Thư bác sĩ xác nhận范围 consultation, không phải thiết bị chẩn đoán.

**H: Nếu hệ thống báo nhầm thì sao?**
Đ: 4-Layer Defense (FPR mục tiêu <5%) + cảnh báo 3 mức; WARNING chỉ khuyên
kiểm tra, CHỈ EMERGENCY mới kêu gọi gọi 115. Design fail-safe: im lặng/không
dữ liệu KHÔNG BAO GIỜ tạo cảnh báo.
