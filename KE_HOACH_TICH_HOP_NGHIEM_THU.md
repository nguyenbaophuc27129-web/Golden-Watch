# KẾ HOẠCH CHỦ — TÍCH HỢP, KIỂM ĐỊNH, NGHIỆM THU (08/09 → CẤP QUỐC GIA)

> Cập nhật 07/09/2026 sau khi: gait v2 LOSO xong (AUC 0.884), face M1-06 đột phá
> (AUC 0.845 block-CV), app tách 2 phần Camera/Radar xong, bộ metrics chuẩn AI
> + y tế chạy xong (`metrics_pack_20260907_223217`, `BANG_THANG_DO_KIEM_DINH.md`).
> Mốc thi: **hồ sơ 05–10/10/2026 · TP 04–25/11/2026 · chọn đội quốc gia 05–07/12/2026**.

## ĐỌC HIỂU HIỆN TRẠNG (3 dòng)
- **Đã nối app thật**: face rules, arm (YOLO+ML synthetic), gait camera (rule), speech (TORGO median-3), radar (SIM + COM port thật), fusion → defense → NIHSS → báo động → QR/PDF.
- **Đã train số liệu thật, chưa nối app (cố ý, trung thực)**: face ML 5-feat (AUC 0.845) — chờ protocol B; gait v2 PhysioNet (AUC 0.879) — sai khác không gian đặc trưng với camera (M4-07), chỉ dùng khi có cảm biến lực.
- **Thiếu số liệu (ghi THIẾU, không bịa)**: arm trên data thật, NIHSS hồi quy (chờ 50 video L-01A), tiếng Việt speech (L-02), radar hardware (08/09).

---

## GIAI ĐOẠN 1 — PHẦN CỨNG RADAR + VỮNG HÓA APP (08–14/09)

| Ngày | Việc | Tiêu chí xong | Ghi logbook |
|---|---|---|---|
| **08/09** | Test radar thật `module5/test_radar_hardware.py`: 10 kịch bản (đi bộ, ngã lên nệm, bất động, 2 người…) | ≥8/10 pass; xác nhận baud 256000, COM port nào; đo độ trễ phân tích | M5-02 + bảng kịch bản |
| 09/09 | Vá lỗi radar phát hiện ngày 08/09 (nếu có); nối `RadarModule.list_ports()` đã có trong tab 📡 Radar vào demo thật | Kịch bản ngã thật → DANGER ≤5s | M5-02b |
| 09–10/09 | Regression test SYS-14: script chạy detector face trên ảnh mặt thật phải trả `status ≠ NO_DETECTOR` (chặn stub .task tái xuất) | Test vào `metrics_suite` (58→59 test) | SYS-14b |
| 10–11/09 | Smoke test app 2 phần mới: camera 30 phút liên tục không crash; radar SIM chuyển LIVE; 58/58 unit test lại từ đầu | 0 crash, tất cả tab render | SYS-17 |
| 11–12/09 | Quyết định M4-06: ngưỡng gait runtime 64 vs Youden LOSO — **khuyến nghị: giữ 64 cho camera (rule-space khác PhysioNet)**; ghi quyết định + lý do | Mục quyết định trong LOI_SO_MODULE | M4-06 đóng |
| 13/09 | Lấy URL citation Kaggle face dataset (quota web reset) → cập nhật `face_dataset_info.json` + BANG_CO_SO_KHOA_HOC | Citation đầy đủ | M1-08 |
| 14/09 | **Nghiệm thu nội bộ #1** (chạy KICH_BAN_NGHIEM_THU.md S0–S3) | Pass S0–S3 | NT-01 |

## GIAI ĐOẠN 2 — KIỂM ĐỊNH KHOA HỌC CỐT LÕI (15–30/09)

| Ngày | Việc | Tiêu chí xong | Logbook |
|---|---|---|---|
| 15–18/09 | **L-01A**: tải/ selección 50 video công khai có NIHSS chuẩn (YouTube kênh stroke education, PhysioNet nếu có) → chấm tay 4 items Golden-Watch đo | Bảng 50 dòng: NIHSS bác sĩ vs máy | SYS-18 |
| 19–20/09 | Tính **MAE/RMSE/R² + Bland–Altman + weighted kappa** máy-vs-chuẩn trên 50 video → cập nhật BANG_THANG_DO_KIEM_DINH mục 2 từ THIẾU→có số | MAE ≤ 2 điểm NIHSS mục tiêu | SYS-18b |
| 21–23/09 | **Protocol B (SYS-15)**: 100 tình huống người khỏe (cười, ngáp, quay nghiêng, ánh sáng yếu, đeo khẩu trang, đi chậm người già…) chạy app thật → tỷ lệ báo giả | EMERGENCY giả ≤5%; tổng báo giả ≤15% | SYS-15b |
| 24–25/09 | Quyết định M1-07 cuối: nếu protocol B đạt → nối face ML 5-feat vào app (artifact JSON đã có) với feature-flag; không đạt → giữ rules + ghi rõ | Flag `FACE_ML_ENABLED` + A/B số liệu | M1-07 đóng |
| 26–27/09 | **L-02**: thu 20–30 mẫu tiếng Việt (đội + người thân đọc câu chuẩn, nói rõ/nói líu, người cao tuổi) → test speech module trên tiếng Việt | Bảng per-sample prob; xác nhận Vosk-vn chạy | M2-11 |
| 28–30/09 | Robustness matrix camera: 6 điều kiện ánh sáng × 3 khoảng cách × 3 góc quay (dùng chính đội) → bảng S/X | Không cột nào sập hoàn toàn; ghi biên giới thao tác | SYS-19 |

## GIAI ĐOẠN 3 — HỒ SƠ DỰ THI (01–10/10) ⚠️ DEADLINE 05–10/10

| Hạn | Việc |
|---|---|
| 01–02/10 | Sách cậy đính chính: D-01→D-13 (số liệu thống nhất về 1 nguồn: metrics_pack mới nhất); khớp TONG_QUAN + slides + BANG_CO_SO_KHOA_HOC với số thật (0.845/0.879/0.992, THIẾU ghi rõ) |
| 02/10 | **Prototype vật lý** (tăng điểm Chế tạo 20đ): vỏ/hộp gắn tường hoặc chân đế chứa LD2450 + camera + mạch, màn hình demo — KH được chấm "chế tạo", chỉ app laptop = điểm yếu | Vỏ in 3D/gỗ + lắp xong, chạy được S1–S4 |
| 03/10 | Video < 3 phút: kịch bản 3 hồi (bình thường → dấu hiệu → cấp cứu + QR bàn giao), quay màn hình app thật + prototype, không dàn diễn viên bệnh nhân |
| 03/10 | **Poster online theo Phụ lục 3** (bắt buộc nộp, công bố website BTC) — KH 5.2.2 |
| 04/10 | Phụ lục 1 — KHAI BÁO AI (bắt buộc): liệt kê Claude/AI dùng ở đâu (code support, dữ liệu công khai), phần tự làm; Phụ lục 2 — logbook (LOI_SO_MODULE là nguồn); **KHAI BÁO KẾ THỪA** nếu có dùng kết quả từ dự án/cuộc thi khác (fga_project) — KH 5.2.2 "phải cung cấp đầy đủ thông tin" |
| 05/10 | **NỘP HỒ SƠ** + GVHD ký kế hoạch nghiên cứu (bắt buộc TRƯỚC khi nghiên cứu — kiểm tra đã ký) + repo Git chuyển PRIVATE |
| 06–10/10 | Buffer + in ấn bản cứng theo checklist thể lệ |

## GIAI ĐOẠN 4 — THI CẤP TRƯỜNG & TP (10/10 → 25/11)

| Tuần | Việc |
|---|---|
| 11–17/10 | Diễn tập KICH_BAN_NGHIEM_THU 3 vòng với vai: người bệnh/người thân/hội đồng; gĩ tốc độ trả lời Q&A (dùng phần "hỏi-đáp" Lỗi_ngày + SYS-15) |
| 18–25/10 | Poster + demo box (radar gắn nệm, camera laptop); dự phòng: video demo offline nếu wifi lỗi |
| **Thi trường** | Chạy S1–S5 trực tiếp; hội đồng thử S5 (cười/ngáp) — phải qua |
| 01–25/11 | Vòng TP: nâng cấp trình bày (35đ = trọng số lớn nhất); thống kê câu hỏi hội đồng trường → chuẩn bị sẵn |

## GIAI ĐOẠN 5 — QUỐC GIA (05–07/12)
- Chốt bộ số liệu lần cuối + bảng so 3 môi trường (nhà/case của trường/hội trường).
- Đóng gói repo archive + data provenance (Kaggle/PhysioNet/TORGO citation) — chuẩn TRIPOD+AI.

---

## ⚠️ CHECKLIST TUÂN THỦ KH (vi phạm = KHÔNG ĐƯỢC CHẤM)

- [ ] GVHD **ký phê duyệt kế hoạch nghiên cứu TRƯỚC khi nghiên cứu** (KH 3.2)
- [ ] Thời gian nghiên cứu: từ **tháng 01/2026** đến trước ngày nộp — sổ nhật ký phải khớp
- [ ] Dự án **chưa công bố** ở cuộc thi khác; nếu kế thừa (fga_project) khai đầy đủ trong báo cáo (KH 5.2.2)
- [ ] Báo cáo ≤15 trang A4, Times New Roman 14, lề 3/2/2/2 cm, dòng đơn, **KHÔNG ghi tên đơn vị**
- [ ] Cấu trúc báo cáo đúng KH: trang bìa → tóm tắt (tính mới/khoa học/thực tiễn/cộng đồng) → lý do → vấn đề+giả thuyết → thiết kế&PP (kèm **rủi ro + an toàn**) → tiến hành + kết luận → **tài liệu tham khảo ≥5**
- [ ] Sổ nhật ký theo Phụ lục 2 (chuyển từ LOI_SO_MODULE)
- [ ] Khai báo AI theo Phụ lục 1
- [ ] Poster online theo Phụ lục 3
- [ ] Video sản phẩm < 3 phút, nộp trực tuyến theo mẫu Sở
- [ ] Nộp hồ sơ trực tuyến **05–10/10/2026** + in bản giấy
- [ ] Học sinh đạt hạnh kiểm Khá trở lên năm học 2025–2026; mỗi em chỉ 1 dự án

## 📊 CHẤM ĐỂ BÀN HIỆN TẠI vs MỤC TIÊU (đối chiếu KH 5.3.2b)

| Tiêu chí | Max | Hiện tại | Sau G1–G5 |
|---|---|---|---|
| Vấn đề nghiên cứu | 10 | 8,5–9 | 9 |
| Thiết kế & phương pháp | 15 | 11–12 | 13–14 |
| Chế tạo & kiểm tra | 20 | 10–12 | 15–17 |
| Tính sáng tạo | 20 | 13–14 | 15–16 |
| Trình bày | 35 | 15–18 | 28–31 |
| **Tổng** | 100 | **≈58–66 (KK–Ba TP)** | **≈83–89 (Nhất TP → có cơ hội QG)** |

**Mốc giải nhì quốc gia ≈ 90+ điểm** tại vòng tuyển chọn. Ba thứ quyết định: kiểm tra thật (SYS-18/15b), prototype vật lý, trình bày 35đ.

---

## DANH SÁCH MỞ (mọi item đã gán giai đoạn)

| ID | Item | Giai đoạn |
|---|---|---|
| M5-01/02 | Radar thật 10 kịch bản | G1 |
| SYS-14b | Regression stub .task | G1 |
| SYS-17 | Smoke app 2 phần + 58/58 | G1 |
| M4-06 | Ngưỡng gait | G1 |
| M1-08 | Citation Kaggle | G1 |
| L-01A/SYS-18 | 50 video NIHSS → MAE/RMSE/R²/Bland–Altman/wκ | G2 |
| SYS-15b | Protocol B 100 tình huống | G2 |
| M1-07 | Nối face ML (có điều kiện) | G2 |
| L-02/M2-11 | Tiếng Việt speech | G2 |
| SYS-19 | Robustness ánh sáng/góc | G2 |
| D-01→D-13 | Đính chính hồ sơ về 1 nguồn số | G3 |
| PR-01 | **Prototype vật lý** (vỏ + lắp đặt) | G3 (02/10) |
| PR-02 | **Poster online Phụ lục 3** | G3 (03/10) |
| PR-03 | Khai báo kế thừa fga_project (nếu có) trong báo cáo | G3 (04/10) |
| Phụ lục 1/2, video, repo private, GVHD ký | Hồ sơ | G3 |
| NT-01→NT-03 | Nghiệm thu nội bộ | G1/G3/G4 |
