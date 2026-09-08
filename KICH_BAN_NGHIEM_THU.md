# KỊCH BẢN NGHIỆM THU CUỐI CÙNG — TRƯỚC THI CẤP TRƯỜNG (v1.0 — 07/09/2026)

> Mục đích: chứng minh hệ thống **hoạt động thật, đúng kịch bản, không dựng**.
> Cách chạy: đúng thứ tự S0→S6, mỗi mục có người bấm giờ + người ghi. Fail bất
> kỳ mục ⛔ thì KHÔNG đem đi thi — sửa rồi chạy lại toàn bộ.
> Kết quả ghi vào `test_results/nghiem_thu/NT_<ngay>.md`.

## S0 — KIỂM TRA TRƯỚC CHẠY (Pre-flight, ⛔ bắt buộc pass hết)

| # | Hạng mục | Tiêu chí PASS | Cách kiểm |
|---|---|---|---|
| 0.1 | Unit test | 58/58 (hoặc hơn nếu thêm SYS-14b) | `pytest src -q` thoát 0 |
| 0.2 | Model files | `face_landmarker_v2.task` ≥3MB (không phải stub 9KB); `arm_weakness_*.pth`; `speech_torgo_*.pth`; `vosk-model-vn-0.4` | `ls -la models/` |
| 0.3 | Camera | Ảnh preview rõ, 640×480 | mở app, xem tab 📷 |
| 0.4 | Radar | Tab 📡 hiện COM port thật (hoặc SIM nếu chưa cắm); baud 256000 | `RadarModule.list_ports()` |
| 0.5 | Disclaimer | "KHÔNG PHẢN CHẨN ĐOÁN — GỌI 115" hiển thị cả sidebar + header | nhìn màn hình |
| 0.6 | Còi | Test còi phát tiếng rõ | nút Test còi |
| 0.7 | Bộ số liệu | `metrics_pack_*` + `BANG_THANG_DO_KIEM_DINH.md` có mặt (đối chiếu khi hội đồng hỏi) | mở file |

## S1 — NGƯỜI BÌNH THƯỜNG (expected: NORMAL)

| Bước | Hành động | Tiêu chí PASS |
|---|---|---|
| 1.1 | Nhập thông tin người bệnh (sidebar) | Lưu được |
| 1.2 | Người A ngồi nói chuyện bình thường 60s (bật giám sát) | Điểm nguy cơ < 30, trạng thái NORMAL, **không** tin nhắn báo động trong tab 🚨 |
| 1.3 | Đi bộ ngang camera 15s | Gait không DANGER kéo dài; radar LIVE: không ngã |

## S2 — MÔ PHỎNG DẤU HIỆU ĐỘT QUỴ (expected: WARNING/EMERGENCY + 115)

| Bước | Hành động | Tiêu chí PASS |
|---|---|---|
| 2.1 | Người A: cố tình méo một bên miệng + nhắm mắt một bên (mô phỏng méo mặt) 30s | Face prob tăng ≥ 30; NIHSS item4 ≥ 1 trong tab 📊 |
| 2.2 | Trong lúc đó đọc câu không rõ ràng (lắp bắp lầm, mô phỏng nói khó) qua 🗣️ Kiểm tra nói | Speech prob ≥ 30 → DÙNG median-3; NIHSS item10 ≥ 1 |
| 2.3 | Kết hợp 2.1+2.2 | Fusion ≥ WARNING; tab 🚨 có **tin nhắn báo động kèm việc cần làm NGAY**; còi kêu; nhắc GỌI 115 |
| 2.4 | Tay giơ yếu (một tay rơi xuống) 20s | Arm prob tăng; NIHSS item5 ≥ 1 |

## S3 — RADAR PHÁT HIỆN NGÃ (expected: DANGER + báo người thân)

| Bước | Hành động | Tiêu chí PASS |
|---|---|---|
| 3.1 | Radar LIVE: người đi lại quanh phòng | Tab 📡: dịch chuyển cập nhật, trạng thái NORMAL/WARNING |
| 3.2 | NGÃ AN TOÀN lên nệm/gối dày (có người đỡ bên cạnh!) | Bất động > 45s hoặc dịch chuyển đột biến > 100cm → radar DANGER; fusion tăng; nếu đạt ngưỡng → EMERGENCY |
| 3.3 | SIMULATION: chuyển scenario=fall trong tab 📡 (demo khi không cắm radar) | Cùng kết quả DANGER — chứng minh logic không phụ thuộc hardware |
| 3.4 | Audio-gate: bật speech WARNING rồi ngã | Cảnh báo củng cố (AND-gate ×0.3) |

## S4 — BÀN GIAO BỆNH VIỆN (Handoff)

| Bước | Hành động | Tiêu chí PASS |
|---|---|---|
| 4.1 | Sau S2/S3: tab 🏥 → Tạo báo cáo QR + PDF | PDF tải được; QR quayscan bằng điện thoại hiện tóm tắt |
| 4.2 | Kiểm nội dung PDF | Có: thông tin người bệnh, NIHSS ± CI từng item, triage + bệnh viện đề xuất + SĐT, log sự kiện T1/T3, stats defense |
| 4.3 | Triage | Không gọi "chẩn đoán"; có disclaimer |

## S5 — CHỐNG BÁO GIẢ (hội đồng sẽ thử — PHẢI QUA)

| Bước | Hành động | Tiêu chí PASS |
|---|---|---|
| 5.1 | Cười lớn / ngáp / cau mày 30s | **KHÔNG** EMERGENCY (defense L2/L4 + face tĩnh giữ ổn định) |
| 5.2 | Quay đầu góc nghiêng, ra vào khung hình, 2 người trong hình | Không spike điểm kéo dài; NO_FACE xử lý êm (không crash) |
| 5.3 | Vẫy tay nhanh trước camera (chuyển động mạnh) | L2 context không đẩy lên EMERGENCY |
| 5.4 | Radar SIM scenario=normal + mọi người bình thường 5 phút | 0 báo động; trend phẳng < 30 |

## S6 — FAIL-SAFE (mất thiết bị phải AN TOÀN)

| Bước | Hành động | Tiêu chí PASS |
|---|---|---|
| 6.1 | Rút camera giữa chừng | Thông báo lỗi rõ, **không** phát EMERGENCY, app không crash |
| 6.2 | Rút radar (LIVE→mất serial) | Fallback SIMULATION + cảnh báo chế độ, không ngừng fusion phần camera |
| 6.3 | Tắt mic khi speech đang chạy | Báo lỗi nhẹ, các module khác vẫn chạy |
| 6.4 | Tắt监控 giữa chừng → bật lại | Trạng thái reset sạch (history radar xoá), không kế thừa cảnh báo cũ |

## SAU NGHIỆM THU

1. Xuất `NT_<ngay>.md` + ảnh chụp từng màn hình chính (NORMAL / WARNING / QR).
2. Chống đối số "dàn dựng": mọi kịch bản đều làm **tại chỗ, trước người chứng kiến**; số mô phỏng radar có nhãn SIM rõ ràng; số ML đối chiếu `metrics_pack` (AUC/CI) — không trình bày số train.
3. 3 câu hỏi hội đồng phải trả lời được (script SYS-15 trong LOI_SO_MODULE.md):
   - "Không có bệnh nhân thì kiểm chứng thế nào?" → 50 video NIHSS chuẩn + protocol B + thang đo chuẩn BANG_THANG_DO_KIEM_DINH (TRIPOD+AI/STARD).
   - "Số này train hay test?" → mọi số là out-of-fold/LOSO + CI.
   - "Báo giả thì sao?" → S5 + tỷ lệ protocol B ≤15%, EMERGENCY giả ≤5%.

## TIÊU CHÍ "SẴN SÀNG THI TRƯỜNG" (tất cả ✅ mới đem đi)

- [ ] S0 7/7 · S1 3/3 · S2 4/4 · S3 4/4 · S4 3/3 · S5 4/4 · S6 4/4
- [ ] Thời gian demo trọn S1→S4 ≤ 10 phút (tập trước 3 lần)
- [ ] Máy thi: cài sẵn venv + model, tắt update Windows, sạc đầy, có monitor thứ 2 cho người thân-view
- [ ] Phương án dự phòng: video demo 3 phút offline nếu camera phòng lỗi
