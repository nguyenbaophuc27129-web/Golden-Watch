# 🔴 LỖI_NGÀY — DANH MỤC LỖ HỎNG & VIỆC KHẮC PHỤC (đánh giá độc lập 07/09/2026)

> Đánh giá theo tiêu chí chấm KHKT TP (TT06/2024): Vấn đề 10đ | Thiết kế & PP 15đ |
> **Thực hiện: chế tạo & kiểm tra 20đ** | Sáng tạo 20đ | **Trình bày 35đ**.
> Trạng thái ngày 07/09: **không có 1 chỉ số hiệu năng nào của hệ thống tích hợp được đo trên dữ liệu thật** → đang mất phần lớn 20đ "Thực hiện".
> Deadline hồ sơ: **05–10/10/2026** | Thi TP: 04–25/11 | Vòng tuyển quốc gia: 05–07/12.

---

## MỨC ĐỘ ƯU TIÊN
- **P0 — CHÍ MẠNG:** không khắc phục → không vượt vòng TP / bị bắt khi phản biện
- **P1 — NÂNG CẤP:** ảnh hưởng điểm số đáng kể
- **P2 — DỌN DẸP:** tính nhất quán tài liệu, lợi thế trình bày

---

# P0 — LỖ CHÍ MẠNG

### L-01: Chưa có validation đầu-cuối của hệ thống tích hợp trên dữ liệu thật 🔴
- **Bằng chứng:** test 58/58 = unit-test logic (đúng vai trò nhưng KHÔNG phải performance); `test_results/` chỉ có số liệu module 2 (TORGO) + metrics_suite; các mục tiêu Sens>90%/Spec>95%/FPR<5% (README, TONG_QUAN, LỊCH) là **mục tiêu chưa đo**.
- **Hậu quả thi:** mất 20đ "Thực hiện"; giám khảo y sinh hỏi "chứng cứ lâm sàng đâu?" → không có câu trả lời.
- **Khắc phục (protocol không cần bệnh viện — hợp pháp 100%):**
  1. **A — NIHSS correlation (50 video công khai):** video NIHSS chuẩn có điểm công bố; 2 rater nội bộ tự đào tạo, chấm độc lập (đo inter-rater); chạy pipeline → r, MAE (`validation_metrics.py` đã có). Mục tiêu r≥0.85, MAE<2.
  2. **B — 100+ kịch bản tại nhà trên người khỏe mạnh tự nguyện (HS + người thân):** nhóm FPR: cười/ngáp/tập/nói/nhiễu/đi bộ (không bệnh nhân → hợp pháp); nhóm TPR: mô phỏng thiếuнолет có kiểm soát (lép mặt chủ động, nói líu, tạ nhẹ 1 tay) — ghi rõ "simulated deficits on healthy volunteers, có đồng ý từng người".
  3. **C — Robustness matrix môi trường:** ánh sáng 10/50/300 lux × khoảng cách 0.5/1/3m × kính/không kính × noise 40/60/80dB → bảng % hoạt động (trả lời trước câu hỏi "ánh sáng/không khí").
  4. **E — 2 thư chuyên môn dạng consultation/review:** bác sĩ phòng khám tư / nghỉ hưu / giảng viên (không phải hợp tác nghiên cứu → không sinh thủ tục; mẫu sẵn `MAU_THU_XAC_NHAN_BAC_SI.md`), kèm feedback 10 case.
  5. **F — Benchmark:** FPS/latency trên RTX3050; radar LD2450 thật (10 kịch bản: ngã xuống đệm, đi, ngồi, 2 người).
  6. **G — Handoff timing đo thật:** 10 người giả lập tiếp nhận, có vs không có QR report → ra con số thật thay "15→5 phút".
- **KPI xong:** 1 file `validation_report_final.pdf` + số liệu Sens/Spec/CI thật.

### L-02: Mismatch domain mô hình Speech — train tiếng Anh (TORGO), triển khai tiếng Việt 🔴
- **Bằng chứng:** `speech_torgo_20260828_211130.pth` train trên TORGO (dysarthria tiếng Anh, nguyên nhân bại não/ALS — không phải đột quỵ); đối tượng thật = người già VN.
- **Câu hỏi phản biện chắc chắn:** "Model tiếng Anh phát hiện gì trên tiếng Việt?"
- **Khắc phục:** (a) giữ TORGO = bằng chứng phương pháp; lập luận bảo vệ: jitter/shimmer/pause-rate là đặc trưng âm học language-agnostic (có literature); (b) **thu dataset nội bộ tiếng Việt nhỏ (20–30 người khỏe): câu chuẩn vs mô phỏng nói líu** → cross-domain test bắt buộc; nếu đủ dữ liệu thì fine-tune lại; (c) công bố cả 2 số liệu + hạn chế rõ ràng. Câu trả lời chuẩn: "huấn luyện trên dữ liệu công khai tiếng Anh, kiểm chứng chuyển đổi trên dữ liệu tiếng Việt nội bộ, hạn chế được ghi nhận".
- **Lưu ý thêm:** dysarthria ≠ đột quỵ — Parkinson v.k. cũng gây báo nhầm → ghi nhận hạn chế đặc hiệu.

### L-03: Ngưỡng chọn trên chính tập test (optimistic bias) 🔴
- **Bằng chứng:** Youden J=1.00, th=56%, TPR 96.3%/FPR 0% trên đúng 55 session dùng để đánh giá (LOI_SO M2-11). FPR 0% + J=1.00 trên n=55 là dấu hiệu overfit trực diện với giám khảo có kinh nghiệm thống kê.
- **Khắc phục:** chia train/test (chọn ngưỡng trên nửa A, đánh giá trên nửa B); báo Wilson CI cho TPR/FPR (n nhỏ); báo n **người**, không phải n file; công bố CI. Vẽ ROC + bootstrap CI.

### L-04: Số liệu "độ chính xác" trên dữ liệu mô phỏng/không thể bảo vệ 🔴
- **Bằng chứng:** Arm 99.50% (dữ liệu mô phỏng — README thừa nhận), Gait 86.13% trên n≈10 (M4-03). Face 93.75% "n=2783 frames tự thu" nhưng `training/train_face_model.py` trỏ tới dataset công khai "Annotated stroke and non stroke Dataset" → **mâu thuẫn nguồn dữ liệu**; và n frames ≠ n người (pseudo-replication).
- **Khắc phục:** (a) arm/gait: gỡ mọi con số mô phỏng khỏi báo cáo/poster; thay bằng benchmark kịch bản thật (cùng protocol B); (b) face: xác định rõ nguồn (công khai → trích dẫn đúng; tự thu → n người thật là bao nhiêu), tính lại accuracy **theo người** (leave-person-out nếu có thể); (c) mọi bảng số liệu kèm n và CI.

### L-05: NIHSS mapping không đúng quy trình chấm chuẩn của item tương ứng 🔴
- **Bằng chứng:** NIHSS item 5 (motor arm) = drift test 2 tay duỗi, item 6 (motor leg) = drift khi nằm ngửa — KHÔNG phải đo khi đi bộ (gait) hay swing tay khi đi. Code `nihss_estimator.py` map gait→item6, arm-walking→item5.
- **Hậu quả:** giám khảo y khoa: "item 6 chấm khi nằm, sao em dùng đi bộ?"
- **Khắc phục (rẻ, hiệu quả cao):** đổi tên toàn bộ thành **"NIHSS-derived deficit score (4 items, max 13)"** — công cụ ước tính mức thiếu hụt theo *thang điểm NIHSS*, KHÔNG phải chấm NIHSS chuẩn; nói rõ bác sĩ mới chấm NIHSS thật; validation Phần A chính là bằng chứng tương quan. Nếu muốn mạnh hơn: thêm chế độ "test chủ động" (vươn 2 tay 10s như drift test) — đúng quy trình NIHSS hơn và dễ đo hơn gait.

### L-06: Thiếu giấy tờ thi theo thể lệ 🔴
- **Bằng chứng:** thể lệ 6756: (1) Sổ nhật ký trình bày khi phỏng vấn — LOI_SO_MODULE.md tốt nhưng phải chuyển đúng **Phụ lục 2** (mẫu của Sở, đầy đủ từ ý tưởng→sản phẩm, có GVHD ký); (2) **khai báo sử dụng AI tạo sinh theo Phụ lục 1 — BẮT BUỘC**, dự án phụ trợ AI nặng phải khai trung thực phạm vi sử dụng; (3) GVHD phải **ký duyệt kế hoạch nghiên cứu TRƯỚC khi nghiên cứu** (dự án bắt đầu ~cuối 08/2026 — kiểm tra đã ký chưa, thiếu thì lập tức bổ sung); (4) **video sản phẩm < 3 phút** (thể lệ) — kế hoạch đang ghi 3–5 phút → sửa; (5) báo cáo ≤15 trang, không ghi tên đơn vị; (6) poster online theo Phụ lục 3; (7) thời gian nghiên cứu phải nằm trong khung 01–10/2026 (đạt) và "dự án chưa công bố ở cuộc thi khác" → **GitHub repo nên chuyển private** đến hết thi (tránh mờ nhược "công bố" + chống sao chép).

---

# P1 — LỖ NÂNG CẤP

### L-07: Arm đo bằng pixel 🔴 (đã tự ghi M3-03)
- Đo px phụ thuộc khoảng cách camera. **Fix 1 ngày:** chuẩn hóa theo tỷ lệ cơ thể (chia cho khoảng cách 2 vai hoặc chiều dài thân từ keypoints) → hết phụ thuộc khoảng cách.

### L-08: Defense Layer 2 (CONTEXT_SUPPRESS) có thể che đột quỵ thật 🔴
- Suppress hoàn toàn face/arm/gait khi "EXERCISE/WALKING" → người bệnh đang vận động khi khởi phát thì hệ thống câm lặng. **Fix:** thay suppress = kéo dài cửa sổ xác nhận (persistence requirement tăng khi vận động, KHÔNG bao giờ tắt tín hiệu); keep fail-safe ghi trong defense_note.

### L-09: Kịch bản demo phi thực tế (2h sáng NIHSS=12) 🟡
- Đêm: camera tắt, người đang ngủ → không thể NIHSS. **Fix:** viết lại Scenario 1 = ngã/bất hoạt đêm (radar + audio AND-gate → cảnh báo người thân, KHÔNG NIHSS); NIHSS chỉ ban ngày hoặc khi có mặt. Sửa trong TONG_QUAN + script demo.

### L-10: Trọng số fusion tự đặt, chưa phân tích độ nhạy 🟡
- **Fix:** bảng sensitivity analysis (đổi ±0.05 mỗi trọng số → Sens/Spec đổi bao nhiêu) hoặc fit logistic regression nhỏ trên dữ liệu test B → "trọng số học từ dữ liệu" hợp lệ hơn "trọng số tự chọn".

### L-11: Thresholds SELF-DESIGN chưa nguồn (M1-02, M4-04) 🟡
- Face 25%/12°/15°, gait symmetry <15%. **Fix:** search PubMed sau 13/09 (quota hết); nếu không có nguồn → đổi thành "ngưỡng học từ dữ liệu calibration nội bộ + hệ số an toàn", không mượn danh "từ Smith 2023".

### L-12: Radar hardware chưa test thật; baud 256000≠115200 🟡
- **Fix:** 10 kịch bản thật (ngã xuống đệm, đi, ngồi, 2 người cùng phòng — LD2450 track 3 mục tiêu, chưa xử lý multi-target).

### L-13: "Face recognition >95%" trong kế hoạch test Group C — tính năng chưa tồn tại 🟡
- **Fix:** xóa khỏi kế hoạch hoặc ghi "future work"; thay bằng quy tắc "chỉ phân tích người ở trung tâm khung hình".

### L-14: Subtype hint với "% confidence" không có cơ sở xác suất 🟡
- Rule-based +30/+20/−20... xuất ra "65% confidence" = phát minh. **Fix:** bỏ %; đổi thành nhãn hướng dẫn "Ischemic more likely / Hemorrhagic cannot be excluded" — vẫn giữ tính năng, hết lỗ hỏng.

### L-15: Quyền riêng tư & pháp lý chưa được nêu 🟡
- Camera trong nhà người già; QR chứa dữ liệu y tế (Nghị định 13/2023/NĐ-CP). **Fix:** thêm "Privacy Mode (radar+audio only)" và đoạn pháp lý trong báo cáo (định vị sàng lọc hỗ trợ, không phải thiết bị chẩn đoán; kế hoạch tư vấn pháp lý = future work) → **tăng điểm trưởng thành**, không giảm.

### L-16: Cảnh báo "hệ thống offline" chưa có 🟡
- Fail-safe hiện tại: im lặng không báo động (đúng) nhưng người thân không biết hệ thống chết. **Fix:** heartbeat 5 phút → trạng thái "KHÔNG HOẠT ĐỘNG" trên app gia đình.

---

# P2 — DỌN DẸP TÀI LIỆU (mâu thuẫn số liệu giữa các file)

| # | Mâu thuẫn | Ở đâu | Chuẩn hóa thành |
|---|---|---|---|
| D-01 | Mouth asymmetry 12°/15° vs 25% | TONG_QUAN M1 vs BANG_CO_SO sheet2 | chốt 1 đơn vị + 1 giá trị, đánh dấu SELF-DESIGN |
| D-02 | NIHSS item10 "0–3" vs 0–2 | TONG_QUAN M2 output vs nihss_estimator | 0–2 (đúng Brott 1989) — sửa TONG_QUAN |
| D-03 | 2 radar (ngủ+tắm) vs 1 radar | TONG_QUAN vs LOI_SO M5-02 | 1 radar — cập nhật toàn bộ tài liệu + kiến trúc |
| D-04 | "70% người cao tuổi sống một mình" | TONG_QUAN, demo script | SAI thực tế VN — thay bằng số có nguồn (tỷ lệ đến viện trễ, PubMed "prehospital delay stroke Vietnam") |
| D-05 | FAST "Sens 85–95%" | BANG_CO_SO sheet1 | phóng đại — sửa theo Harbison/ABC khi verify (thường 40–86%) |
| D-06 | "Persistent >10min → stroke 99% specificity" | TONG_QUAN L3 | phát minh — bỏ số, ghi mục tiêu cần đo |
| D-07 | "Handoff 15→5 phút" | TONG_QUAN, demo | chưa đo — thay bằng kết quả test G (L-01.6) |
| D-08 | "DICOM-compatible" | TONG_QUAN handoff | SAI — xóa (PDF/JSON không phải DICOM) |
| D-09 | "Cứu 30.000 người/năm" | demo script | bỏ — không căn cứ |
| D-10 | "Chưa ai làm" ×n | TONG_QUAN, Q&A | thay bằng bảng nghiên cứu liên quan có citation (làm sau 13/09): đã có AI-NIHSS, speech-stroke AI, mmWave fall → định vị mới: **hệ tích hợp đầu-cuối đa phương thức giá thấp, offline, bản địa hóa VN, bằng chứng minh bạch** |
| D-11 | Face "tự thu n=2783" vs dataset công khai | README/BANG vs train_face_model.py | xác định nguồn thật, trích dẫn đúng, tính accuracy theo người |
| D-12 | WPM 120–150 "trẻ" | TONG_QUAN M2 | đối tượng là người già — sửa band theo tuổi già + baseline cá nhân |
| D-13 | Video 3–5 phút | LỊCH NGÀY 29 | thể lệ: <3 phút |

---

# 📅 PHÁT HIỆN MỚI NGÀY 07/09 (sau khi lấy dataset fga_project)

### L-17 [P0 — ĐÃ XÁC NHẬN]: Nhãn face ĐẢO NGƯỢC — model 93.75% train trên nhãn sai
- Toàn bộ .txt trong NonStroke có token "1", trong Stroke có token "0" → script cũ đọc token làm nhãn → model cũ học NGƯỢC. Runtime không dùng model này nên demo chưa hỏng, nhưng con số 93.75% PHẢI CẤT khỏi mọi tài liệu ngay.
- **Đã fix:** nhãn theo thư mục; retrain với block-split + mediapipe Tasks API (kết quả chờ).
- **Bài học:** sanity-check nhãn vs folder là bước bắt buộc; thêm unit test.

### L-18 [P1 — ĐÃ FIX]: KHKT thiếu file .task thật (stub 9KB) → face demo chạy FALLBACK không nhận diện
- Đã copy `face_landmarker_v2_with_blendshapes.task` (3.75MB) vào `models/face_landmarker_v2/` + `src/models/face_landmarker.task`.
- Cần thêm unit test: "detector trên 1 ảnh mặt thật phải trả status ≠ NO_DETECTOR".

### L-19 [P1 — ĐÃ FIX]: Gait v1 chia window ngẫu nhiên → leakage subject; đã train lại bằng LOSO trên 15 subject THẬT (PhysioNet)
- **Kết quả LOSO:** Sens 88.24% CI [65.66, 96.71] | Spec 88.28% CI [82.03, 92.55] | AUC 0.884 | **Subject-level 14/15 (93.33%)**
- Model: `models/gait_classifier_v2_20260907_210433.pth` + metadata JSON (citation + limitation Parkinson-proxy).
- Việc treo M4-06: cập nhật ngưỡng runtime (Youden LOSO = 10×100, cũ 64).

### L-20 [P0 — CẬP NHẬT ĐỘT PHÁ 07/09]: Face — dataset CÓ tín hiệu, vấn đề là đại diện đặc trưng
- **Thí nghiệm 3 tầng** (`training/test_face_signal.py`, JSON `face_signal_test_20260907_221456.json`):
  - T1 Cohen's d: mouth_ratio **d=0.759** (tín hiệu thật); nasolabial d=−0.561, eye d=−0.482 (hướng âm = nhiễu nhãn, KHÔNG giải thích lâm sàng); forehead d=−0.07 (vô tín hiệu)
  - T2/T3 GroupKFold(5) THEO BLOCK (ngoài fold): **Logistic AUC 0.846 | MLP AUC 0.842**
  - Tham chiếu: rules tuyến tính 0.638 | MLP 936 tọa độ thô 0.555
- **Kết luận:** model cũ thất bại do đại diện đặc trưng sai (raw tọa độ + lệch lớp), không phải dataset vô tín hiệu; "93.75%" cũ vẫn bị loại (nhãn đảo + leakage)
- **Lộ trình face:** (1) M1-07: train model 5 features + class-weight, ngưỡng Youden trên validation block; (2) chỉ wire runtime SAU protocol B webcam thật xác nhận cùng xu hướng; (3) bằng chứng face chính = protocol B; dataset công khai = benchmark khắc nghiệt + limitation

### Kết quả đã có (bằng chứng thật đầu tiên của dự án)
| Module | Đánh giá | Kết quả |
|---|---|---|
| Gait (M4) | LOSO 15 subject thật PhysioNet | Sens 88.2%, Spec 88.3%, AUC 0.884, subject 14/15 |
| Face rules (M1 runtime) | 776 ảnh test block-split | AUC 0.638, Sens 50.3%, Spec 73.2% |
| Face PyTorch (M1) | block-split + nhãn đúng | AUC 0.555 (~ngẫu nhiên) — KHÔNG dùng, bỏ số 93.75% cũ |
| Radar (M5) | hardware thật | chờ 08/09 (script sẵn `module5/test_radar_hardware.py`) |
| Speech (M2) | TORGO | đã có, chờ data tiếng Việt nội bộ |

---

# THỨ TỰ LÀM ĐỀ XUẤT (đến 05/10)

| Tuần | Việc | Ghi chú |
|---|---|---|
| Tuần 1 (còn lại tháng 9) | L-01A (50 video NIHSS), L-02 (thu data VN nói), L-03 (train/test split), L-05 (đổi tên NIHSS-derived), L-07 (arm chuẩn hóa), D-01..D-13 | song song |
| Tuần 2 | L-01B (100 kịch bản nhà), L-01C (robustness matrix), L-08 (defense), L-10 (sensitivity weights), L-12 (radar thật) | cần 3–5 nhà tình nguyện |
| Tuần 3 | L-01E (2 thư bác sĩ tư/nghỉ hưu), L-01F (benchmark), L-01G (handoff timing), L-16, D-10 (bảng nghiên cứu liên quan) | đặt lịch bác sĩ NGAY tuần này |
| Tuần 4 | Báo cáo ≤15 trang, poster, video <3 phút, sổ nhật ký đúng Phụ lục 2, khai báo AI Phụ lục 1, chuyển repo private, tập phản biện 30 câu | |

# ĐỊNH VỊ TRUNG THỰC
- **Nguyên trạng:** giải KK–Ba TP; rủi ro không vào Top 120.
- **Hoàn thành L-01→L-06 + D:** giải Nhất TP khả thi; đội tuyển quốc gia ~30–40%; **giải nhì quốc gia 15–25%** — phụ thuộc trình bày (35đ) và phản biện.
- Yếu tố quyết định cuối cùng KHÔNG phải thêm công nghệ — là **bằng chứng kiểm định + sự tử tế thống kê + kịch bản demo thực tế**.

---

# 📅 BỔ SUNG 07/09 (ĐÊM) — TÍCH HỢP + BỘ SỐ LIỆU CHUẨN

## L-21: Thiếu bộ số liệu theo thang đo chuẩn AI/y tế → ĐÃ XỬ LÝ ✅
- **Vấn đề:** trước đó chỉ có Sens/Spec/AUC lẻ tẻ; hội đồng có thể hỏi Accuracy/Precision/F1/NPV/Silhouette/CI kèm trích dẫn bài báo.
- **Xử lý:** `training/evaluate_all_metrics.py` → `test_results/metrics_pack_20260907_223217/` (6 kết quả, 13 biểu đồ PNG, Wilson CI + bootstrap AUC CI + Youden + Silhouette + calibration). Đối chiếu trích dẫn: `BANG_THANG_DO_KIEM_DINH.md` (Wilson 1927, Youden 1950, Hanley-McNeil 1982, DeLong 1988, Bland-Altman 1986, Cohen 1960/1968/1988, Hosmer-Lemeshow 1980, Rousseeuw 1987, McNemar 1947, TRIPOD+AI 2024, STARD 2015).
- **Số thật:** face_rules AUC 0.638 · face-ML Logistic 0.845 [0.802-0.885] · gait LOSO 0.879 [0.742-0.961] · speech TORGO 0.992 [0.968-1.0]. Hồi quy NIHSS (MAE/RMSE/R²/Bland-Altman): ghi rõ **THIẾU** chờ L-01A — không bịa số.
- **Trạng thái:** ✅

## L-22: App chưa tách 2 phần Camera/Radar; radar cứng COM3; model gait sai không gian đặc trưng → ĐÃ XỬ LÝ ✅
- **Xử lý:** `app_family.py` v8.1 — 📷 PHẦN 1 (giám sát + nói) / 📡 PHẦN 2 (radar riêng: chọn COM port thật qua `list_ports()`, kịch bản SIM, lịch 120 nhịp + biểu đồ) / phần chung fusion. M4-07: KHÔNG nối gait-PhysioNet vào camera (8 đặc trưng thảm lực ≠ 6 chỉ số pose) — tránh pseudo-science.
- **Kiểm chứng:** py_compile OK + 58/58 PASS (`src/test_all_metrics.py`).
- **Trạng thái:** ✅

## KẾ HOẠCH + NGHIỆM THU (mới)
- `KE_HOACH_TICH_HOP_NGHIEM_THU.md` — lịch 5 giai đoạn 08/09→05-07/12, mọi item mở đã gán ngày.
- `KICH_BAN_NGHIEM_THU.md` — S0 pre-flight → S6 fail-safe, tiêu chí "sẵn sàng thi trường".

## L-23: 3 rủi ro tuân thủ phát hiện khi đối chiếu lại KH cuộc thi (07/09) 🟡
- **Nguồn:** docs/_thele_extract.txt (KH Sở GDĐT, 13 trang) — chấm theo dự án KỸ THUẬT lĩnh vực 5 Kỹ thuật Y Sinh
- **Rủi ro 1 — Poster online Phụ lục 3 (KH 5.2.2):** bắt buộc nộp + công bố website BTC, chưa từng có trong kế hoạch → thêm PR-02 (03/10)
- **Rủi ro 2 — "Chế tạo" chỉ là app trên laptop:** tiêu chí Chế tạo & kiểm tra 20đ đòi hỏi sản phẩm chế tạo thật → thêm PR-01 prototype vỏ máy (02/10); hiện tại hạng mục này ≈10–12/20, mốc giải nhì QG cần 17+
- **Rủi ro 3 — Kế thừa kết quả (fga_project):** KH 5.2.2 "nếu kế thừa kết quả từ cuộc thi khác phải cung cấp đầy đủ thông tin trong báo cáo" → PR-03 khai báo trong báo cáo; không khai = rủi ro loại hồ sơ
- **Điểm đủ tầm giải nhì quốc gia ≈ 90+:** hiện tại ≈58–66 (KK–Ba TP); sau G1–G5 ≈83–89 → phải vượt muốc 17+/20 Chế tạo và 28+/35 Trình bày
- **Trạng thái:** 🟡 đã cập nhật KE_HOACH (checklist tuân thủ + bảng điểm)

---

# 📅 08/09 — BENCHMARK PROTOTYPE + 2 BUG THẬT ĐƯỢC BẮT

## L-24 (M3-06): Arm module CRASH khi khung hình CÓ người 🔴→✅
- **Phát hiện bởi:** `module5/prototype_benchmark.py` (benchmark tự viết phát hiện lỗi — đúng mục đích "kiểm tra" 20đ)
- **Error:** `index 2 is out of bounds for axis 0 with size 2`
- **Nguyên nhân gốc:** docstring nói keypoints (17,3) nhưng YOLO `keypoints.xy` trả **(17,2)** — thiếu trục confidence; `_calculate_arm_metrics` đọc `shoulder[2]` → crash. Trước nay không sập vì webcam chụp cận mặt → YOLO không thấy người → thoát sớm NO_PERSON. **Nguy cơ sập ngay khi demo có người nguyên thân trong hình.**
- **Fix:** `arm_module.py` ghép `keypoints.conf` → `np.column_stack([kp_xy, conf])` (17,3)
- **Trạng thái:** ✅ (chặn lại bằng test 3 trong SYS-14b)

## L-25: Môi trường Python gãy sau nâng torch (numpy 2.5 vs numba) 🔴→✅
- **Triệu chứng:** A9 speech mất `pitch_mean` — "Numba needs NumPy 2.4 or less. Got NumPy 2.5" (pip nâng numpy khi cài torch cu126); thêm torchvision cũ mismatch → `torchvision::nms CUDA backend` lỗi
- **Fix:** pin **numpy==2.3.4** + torchvision 0.29.0+cu126 khớp torch 2.14.0+cu126
- **Bài học:** sau mỗi lần pip install torch phải chạy lại `src/test_sys14b_regression.py` (10/10) + `src/test_all_metrics.py` (58/58)
- **Trạng thái:** ✅ CUDA RTX 3050 6GB hoạt động, suite 58/58

## KẾT QUẢ BENCHMARK PROTOTYPE (SYS-20, số thật 08/09)
- Chu kỳ phân tích ≈ **81 ms** (face 15.3 + arm 27.7 + gait 17.4 + fusion ~0) vs chu kỳ app 3000 ms
- C270 thật: 1280×720 @ **32.2 fps**; GPU CUDA bật thành công (torch cu126)
- Công bố trung thực: CUDA không nhanh hơn CPU ở YOLOv8n 640×480 (overhead transfer) — giá trị GPU là dư địa cho model lớn hơn
- Biểu đồ: `test_results/prototype_benchmark_20260908_050756_latency.png`

## L-26: Streamlit form key trùng session_state key `'patient'` 🔴→✅
- **Phát hiện bởi:** `streamlit.testing.v1.AppTest` (chạy headless app_family.py lần đầu)
- **Error:** `StreamlitValueAssignmentNotAllowedError: st.session_state.patient cannot be modified` — sidebar form key `'patient'` trùng `ss.patient`
- **Fix:** đổi key form thành `'patient_form'`
- **Bài học:** Streamlit 1.55 — widget key KHÔNG được trùng attr session_state; AppTest bắt được loại lỗi này mà chạy tay có thể bỏ qua
- **Trạng thái:** ✅ AppTest chạy lại: 0 exception

## L-27: `ss.cam_error` đọc trước khi khởi tạo 🔴→✅
- **Phát hiện bởi:** AppTest lần 2 (sau khi sửa L-26)
- **Error:** `AttributeError: st.session_state has no attribute "cam_error"` ở lần render đầu tab giám sát
- **Fix:** thêm `ss.cam_error = None` trong `init_session()`
- **Bài học:** mọi attr session_state phải init trong `init_session` — KHÔNG đọc/ghi chỗ khác lần đầu
- **Trạng thái:** ✅ Xác nhận cuối: **0 exception, 6 tab render đủ**, pipeline giám sát chạy thật (alert WARNING 55 từ camera thật)

## XÁC MINH ỔN ĐỊNH 08/09 (mục tiêu "100% trừ radar")
- `py_compile` toàn bộ .py: OK · AppTest: 0 exception · 58/58 unit test · 10/10 pre-flight
- Còn lại duy nhất: **radar phần cứng** — người dùng tự test (module5/test_radar_hardware.py); SIM mode đã chạy được trong app

# 💡 Ý TƯỞNG GHI LẠI (chưa làm — tránh quên)
1. **SYS-15b protocol B:** kịch bản 100 tình huống chống báo giả (vận động/nói chuyện/tv ồn) — chạy để bơm số thật cho H2 trong báo cáo
2. **L-01A 50 video NIHSS chuẩn:** thu + bác sĩ chấm → bơm MAE/RMSE/R²/Bland–Altman/weighted-κ (SYS-18) — hiện đang THIẾU, không được bịa
3. **L-02 tiếng Việt:** thu 20–30 người nói thường + giả nói đớ → speech hiện chỉ mạnh trên tiếng Anh (TORGO)
4. **Eval arm trên dữ liệu thật:** quay 20–30 clip giơ tay (khoẻ/yếu mô phỏng cầm vật nặng 1 tay) — arm là module trọng số cao nhất (.30) nhưng chưa có số eval
5. **Poster online Phụ lục 3 (PR-02):** deadline 03/10 — thiết kế sớm từ bộ 5 biểu đồ charts_accuracy
6. **Kế thừa fga_project (PR-03):** khai báo đầy đủ trong báo cáo nếu dùng — không khai rủi ro loại hồ sơ
7. **Trang bị SIM radar mở rộng:** mô phỏng kịch bản ngã thật khi cắm radar (đi/bỏ tay/đứng lên ngồi xuống) trước M5-02
8. **Video <3 phút:** quay theo KICH_BAN_NGHIEM_THU.md 6 bước S0→S6 — dùng lại làm Phụ lục hồ sơ

## L-28: Nghiêng đầu thử (méo mô) không cảnh báo + MediaPipe mất mặt khi nghiêng 🟡→✅ (08/09)
- **Báo cáo bởi người dùng khi tự tay chạy app:** nghiêng đầu giả méo mặt → không cảnh báo được; nghiêng mạnh thì mất hẳn MediaPipe (không hiện gì)
- **Chẩn đoán 2 nguyên nhân:**
  1. Ngưỡng: nghiêng 20° chỉ +20 điểm ((20−10)/10 × 100 × trọng số tilt 0.20) < ngưỡng WARNING 30 → phải nghiêng **≥25°** mới WARNING — không ai biết vì **không có HUD điểm live** trên khung camera
  2. Mất mặt: `min_face_detection_confidence=0.5` quá gắt → nghiêng >~30° MediaPipe bỏ mặt → NO_FACE score 0
- **Fix (face_module_v7.py):** hạ 3 confidence 0.5→**0.3**; thêm **HOLD 3s** — mất mặt thoáng qua vẫn giữ kết quả gần nhất (đánh dấu `hold: True` trung thực, quá 3s mới trả NO_FACE)
- **Fix (app_family.py):** thêm **KHUNG ĐO CHÍNH XÁC (HUD)** `draw_face_hud()` vẽ lên khung sau khi module chạy xong: oval hướng dẫn đặt mặt + hộp quanh mặt thật từ 478 landmarks + thanh điểm méo mặt có vạch ngưỡng 30 + dòng "nghieng dau: X do (WARNING khi ~>=25)" + metric vượt ngưỡng + chữ MAT FACE khi mất mặt thật
- **Về câu hỏi "YOLO?":** YOLOv8n-pose chỉ có 5 điểm đầu mặt — KHÔNG thay được MediaPipe (cần 478 landmark tính 5 tỉ lệ y khoa). YOLO chỉ dùng được cho tay/dáng đi
- **Kiểm chứng:** khung trống → NO_FACE hold=False đúng; SYS-14b 10/10; suite 58/58
- **Trạng thái:** ✅ — cách test: nghiêng đầu chậm, xem HUD; nghiêng ≥25° → méo mặt WARNING; nghiêng mạnh qua ngưỡng MediaPipe → hiện [HOLD] ≤3s rồi mới báo MẤT MẶT

## L-29 (GÓP Ý NGƯỜI DÙNG): App "chụp rồi load" không phải camera giám sát 🟡→✅ (08/09)
- **Góp ý gốc:** "Vẫn chưa ổn. Khung phải đa dạng MediaPipe VÀ YOLO — là camera giám sát đặt trên cao quay thời gian thực; mở app phải quan sát chuyển động như camera giám sát thông thường, có bất thường thì báo/gửi tin nhắn trên app. Chụp - tính - load liên tục thì không hiệu quả."
- **Nguyên nhận đúng:** v8.1 hiển thị 1 khung/3 giây (slideshow) — không phải video trực tiếp
- **Refactor kiến trúc (app_family.py v8.2):**
  1. **Thread `_camera_worker`** — đọc webcam LIÊN TỤC ~30fps vào buffer có lock; fragment chỉ lấy frame mới nhất
  2. **`video_fragment` (0.15s ≈ 7 khung/giây)** — VIDEO TRỰC TIẾP: vẽ **khung xương YOLO từng khung** (đúng yêu cầu đa dạng YOLO, tối đa 3 người) + **MediaPipe méo mặt + HUD** từ chu kỳ phân tích
  3. **`monitor_fragment` 3s→2s** — chỉ PHÂN TÍCH (face MediaPipe + arm/gait ML + defense + fusion + alert), KHÔNG hiển thị nữa
  4. **Thread `_radar_worker` riêng** — radar quét 0.6s KHÔNG chặn video nữa; đổi radar trong tab 📡 swap ngay vào thread + đóng serial cũ (tránh kẹt COM)
  5. `yolo_lock` — 1 model YOLO dùng chung 2 thread an toàn
- **Kiểm chứng AppTest (BẬT giám sát thật + chờ thread):** 0 exception · cam frame (480,640,3) đổ liên tục · radar result NORMAL · history 2 dòng · fusion 55 WARNING + alert bắn
- **Trạng thái:** ✅ 10/10 + 58/58. Còn lại: MediaPipe mặt vẫn cache 2s (mặt ít chuyển động — chấp nhận được); idea: chạy MediaPipe cũng theo khung nếu cần mượt hơn
9. **MediaPipe mặt theo từng khung:** hiện cache 2s (đủ vì mặt ít chuyển động); nếu muốn mượt như YOLO skeleton → chạy MediaPipe trong video fragment, nhưng detect_for_video VIDEO-mode cần timestamp tăng đơn điệu giữa 2 thread — phải gom về 1 thread gọi hoặc chuyển IMAGE mode

## L-30 (GÓP Ý NGƯỜI DÙNG): Streamlit rerun liên tục → video giật; đề xuất FastAPI ✅ (08/09)
- **Góp ý gốc:** "Vẫn chưa ổn khung hình trên Streamlit — sao phải load liên tục trong khi lấy trực tiếp từ camera? Streamlit làm chậm vậy. Tại sao không dùng FastAPI làm hẳn 1 web trên local — vì mới chỉ là nghiên cứu nên làm local được."
- **Chẩn đoán đúng:** bản chất Streamlit là RE-RUN script mỗi tick (0.15s) + serialize ảnh qua websocket — không bao giờ mượt như video thật; kiến trúc sai bài toán giám sát
- **Giải pháp `web_server.py` (FastAPI + MJPEG — đúng kiểu web camera IP/CCTV):**
  - 3 thread: camera 30fps (vẽ YOLO skeleton + HUD từng khung) · radar riêng · phân tích 2s
  - `/video.mjpg` — MJPEG stream ~15–20fps thật (2.6MB/4s đo được); mở bằng VLC được
  - `/` — dashboard CCTV inline HTML/JS offline: video lớn + trạng thái module + NIHSS + feed báo động + radar bảng + form bệnh nhân + nút test còi + QR/PDF
  - `/api/state` (1s poll) · `/api/radar` POST đổi SIM/LIVE/scenario · `/patient` · `/alert/test` · `/handoff/qr` · `/handoff/pdf`
  - Chạy `python web_server.py` → http://localhost:5001
- **Bugs sửa trong lúc làm:** StreamingResponse (Response không nhận generator); generate_qr trả TUPLE (path, data) — phải unpack; thiếu reportlab+qrcode (chỉ có trong venv cũ) → pip install
- **Bảo mật:** bind **127.0.0.1:5001 duy nhất** — không truy cập được từ máy khác/mạng WiFi; mọi dữ liệu ở lại máy
- **Kiểm chứng:** trụ 200 · state OK (cam online, cycles chạy) · MJPEG 2.58MB/4s · QR 200 (1103B) · PDF 200 (78KB) · alert test OK · đổi radar OK
- **Trạng thái:** ✅ — app_family.py (Streamlit) GIỮ NGUYÊN làm minh chứng so sánh + tab kiểm tra nói (mic)

## L-31 (GÓP Ý NGƯỜI DÙNG 08/09 — buổi 6): Train/MediaPipe/YOLO chưa "kỹ" — thiếu chấm landmark, chưa chống cười/khuất/nằm/gù ✅
- **Góp ý gốc:** "Phần train MediaPipe và YOLO lẽ ra phải có những chấm bao quanh khuôn mặt; khung YOLO có điểm landmark mới đo chính xác sự thay đổi cơ thể. Chỉnh module 1-4; khung phải đo được toàn bộ cơ thể + khuôn mặt linh hoạt; nghe được lời nói khi camera gác cao. Train kỹ hơn với dataset fga_project để phân biệt rõ méo miệng vs miệng thường; tránh: cười lớn, nhech mép thói quen, nằm, khuất nửa người, gù lưng. Ý tưởng: chế độ mở 3 ngày AI quan sát ghi nhận hình thể để tự học đặc điểm riêng."
- **Đã làm (5 việc):**
  1. **Chấm landmark mặt** (239 chấm) vẽ trên khung — cả web_server + app_family
  2. **LAUGH GUARD** (`_detect_expression`): miệng mở + 2 khóe hông lên ĐỐI XỨNG = cười → giảm 60% metric biểu cảm (miệng/rãnh mũi-má/trán), giữ mắt+nghiêng. Test PASS: cười nhận diện, miệng đóng (méo thật) không
  3. **PARTIAL_FACE guard**: bbox landmark chạm mép khung (khuất nửa người/nằm/quay đi) → KHÔNG chấm điểm; fusion+NIHSS coi là missing
  4. **AGC mic xa** (`record_audio`): peak <8% → khuếch đại về 30% (camera gác cao vẫn nghe được)
  5. **CHẾ ĐỘ HỌC 3 NGÀY** (`src/defense/personal_profile.py`): mỗi chu kỳ ghi median 5 chỉ số mặt + gait + motion (KHÔNG lưu ảnh); đủ 3 ngày → baseline cá nhân → `DefenseEngine.calibrate()` bật L4 (vùng xám 30-40 trừ 5 điểm theo cá nhân). Banner xanh trên web hiển thị "Ngày X/3 — n mẫu". File xóa = quét sạch
- **Trung thực về "train kỹ hơn":** dataset fga_project (Kaggle face 3,740) KHÔNG có nhãn "cười/mép-thói-quen/gù" → KHÔNG train được các ca này bằng data hiện có; guards runtime + học 3 ngày là giải pháp đúng; muốn mạnh hơn phải thu dữ liệu có nhãn (bổ sung vào L-01A)
- **Kiểm chứng:** laugh guard PASS · profile self-test PASS · 58/58 + 10/10 · web smoke: MJPEG 1.5MB/3s, profile trong /api/state
- **Trạng thái:** ✅

## L-32: `[WinError 10048]` — bind 127.0.0.1:5001 bị chặn khi chạy server lần 2 🔴→✅ (08/09)
- **Người dùng tự chạy `python web_server.py` và gặp:**
  `ERROR: [Errno 10048] error while attempting to bind on address ('127.0.0.1', 5001): only one usage of each socket address...`
- **Nguyên nhân:** phiên test trước (do trợ lý bật nền để kiểm tra) VẪN đang nghe cổng 5001 → bản mới không bind được. Đã tắt bản cũ → cổng rảnh
- **Fix (web_server.py):** thêm `_port_busy(5001)` kiểm tra trước khi start — nếu cổng bận: in thông báo tiếng Việt + 2 cách tắt (Ctrl+C cửa sổ cũ / `netstat -ano | findstr :5001` + `taskkill /F /PID <pid>`) rồi thoát mã 1 (không còn traceback xấu)
- **Bài học:** script server cần tự kiểm tra cổng trước khi bind; bật nền để test xong phải tắt ngay (quy tắc với trợ lý)
- **Trạng thái:** ✅ cổng 5001 đã rảnh — chạy lại `python web_server.py` bình thường
- **Cải thiện thêm:** chuyển check cổng lên ĐẦU file (trước khi load model MediaPipe/YOLO) — báo lỗi NGAY trong <1 giây thay vì chờ load model ~10s rồi mới báo. Đã test giả lập cổng bận: thoát mã 1 + thông báo đúng

## L-33: HUD còn tư duy "đặt mặt vào khung" — sai với camera giám sát treo cao + khung xương YOLO chưa rõ 🔴→✅ (08/09)
- **Góp ý của người dùng:** (1) "Sao lại đặt mặt vào khung? Dùng API của MediaPipe thì các landmark tự gắn lên khuôn mặt, không phải chạy theo khung như này"; (2) "Vẫn chưa có khung xương trích xuất sâu các điểm của cơ thể trong YOLO?"; (3) "Đã là camera giám sát gia đình thì sẽ đặt lên cao thì làm sao đặt mặt vào được?"
- **Sai sót trong code (vị trí chính xác):**
  1. `draw_face_hud()` — CẢ HAI file `web_server.py` và `app_family.py`:
     - Vẽ **oval hướng dẫn ở GIỮA khung** (`cv2.ellipse(frame, (cx, cy), axes, ...)`) + chữ **"MAT FACE — dat mat vao khung"** → ép người dùng đưa mặt vào giữa khung. Với camera treo cao nhìn xuống (góc người nhỏ, lệch tâm) thì không bao giờ "vào khung" được. MediaPipe FaceLandmarker tìm mặt ở TOÀN ẢNH — oval chỉ là thừa và gây hiểu nhầm
     - **Bug tiềm ẩn nghiêm trọng hơn (chỉ `web_server.py`):** khi bỏ oval ở lần sửa trước, biến `cx, cy` bị xóa nhưng dòng chữ "MAT FACE" vẫn tham chiếu `cx - 160, cy - 6` → nếu mất mặt mà không HOLD, **thread camera sẽ crash NameError**. Đã phát hiện và sửa trong L-33 này
  2. `draw_live_pose()` — cả hai file: `imgsz=320` + `conf=0.30` quá gắt cho người NHỎ/XA từ camera cao; đường xương 2px + chấm 3px khó thấy; không có thông tin đếm người/điểm
- **Fix:**
  1. BỎ hoàn toàn oval + chữ "đặt mặt vào khung" ở cả 2 file. Thay bằng 1 dòng nhỏ khi mất mặt: `MAT: KHONG THAY MAT — landmark tu gan khi co mat` (chân trái khung, không ép vị trí). Landmark MediaPipe + hộp mặt được vẽ NGAY TẠI VỊ TRÍ MẶT THẬT bất kỳ đâu trong khung
  2. `app_family.py` thêm `'PARTIAL_FACE'` vào tập `lost` (bản web_server đã có từ L-31 — 2 file lệch nhau)
  3. `draw_live_pose()` cả 2 file: `conf 0.30→0.25`, `imgsz 320→480` (bắt người nhỏ/xa), đường xương 2→3px, chấm 3→4px, thêm nhãn góc phải: `NGUOI: n  DIEM CO THE: m/(n×17)` — người dùng kiểm chứng được YOLO đang trích xuất đủ điểm cơ thể
- **Kiểm chứng:** `py_compile` OK cả 2 file · 58/58 + 10/10 PASS
- **Bài học:** thiết kế UI phải theo KỊCH BẢN LẮP ĐẶT (camera giám sát treo cao) chứ không theo kịch bản selfie; khi xóa biến phải tìm hết tham chiếu còn lại của biến đó
- **Trạng thái:** ✅

## L-34: Đòi "test 100% chính xác" → HistGB đạt AUC 1.000 và ĐÓ CHÍNH LÀ LEAKAGE 🔴→✅ (09/09)
- **Bối cảnh:** user yêu cầu "cải thiện model tốt 100%, chỉnh thuật toán để test chính xác 100%". v3.1 mở rộng 48 cấu hình (LogReg/SVM-RBF/RandomForest/HistGB/MLP) → **HistGB_200 đạt OOF 0.999 + TEST AUC 1.000, sens 99.3%**
- **Phát hiện (điều tra 2 bước):**
  1. `training/leak_check_blocksize.py` — quét AUC theo cỡ block //50→//200→//500→//1000: LogReg giảm nhẹ 0.942→0.908 (ổn), NHƯNG HistGB vẫn 0.998–1.000 dù chỉ còn 8 block → leak KHÔNG nằm ở frame kề nhau mà do **tên file img_NNNN không mã hoá người/video** → cùng một người nằm cả train lẫn test, mô hình cây NHỚ hình học khuôn mặt từng người (memorization)
  2. **Pose-only check:** chỉ 3 góc yaw/pitch/roll đã đạt AUC 0.627 → 2 thư mục Stroke/NonStroke LỆCH ĐIỀU KIỆN CHỤP (góc/quang) — batch effect. Cây học "vẹt" cả bias này
- **Kết luận trung thực:** AUC 1.000 = đo LEAKAGE + BIAS dataset, KHÔNG phải năng lực phát hiện đột quỵ. Mô hình đáng tin: **LogReg tuyến tính 28 ft — ổn định 0.91–0.94 ở mọi cỡ block**
- **Bài học (ghi vào hồ sơ):** đề bài thi KHÔNG cho phép "tự tạo 100%" — TRIPOD+AI/STARD coi 100% là dấu hiệu lỗi. Giá trị khoa học nằm ở việc TỰ PHÁT HIỆN và CÔNG BỐ leakage (phản biện đánh giá cao)
- **Trạng thái:** ✅ đã biến yêu cầu "100%" thành phát hiện khoa học có bằng chứng

## L-35: ML v3 "đã nối vào app" nhưng KHÔNG BAO GIỜ chấm — lệch tên key artifact ↔ runtime 🔴→✅ (17/09, NK-26)

**Triệu chứng:** smoke 120 ảnh thật qua `process_frame` → 118 ảnh được chấm
điểm nhưng `raw_metrics['ml_prob']` = None 100%. HUD/web luôn hiển thị điểm
rules từ SYS-29 (09/09) — tức là 8 ngày_demo "ML v3" thực chất chạy rules.

**Nguyên nhân:** script train lưu artifact với tên feature generic
(`blend_asym_00..19`, `pose_yaw/pitch/roll`) trong khi runtime phát
`asym_<tên>` + `yaw/pitch/roll`. `FaceMLV3.predict` tra dict theo tên
artifact → thiếu → trả None → fallback rules IM LẶNG (không crash, không
log). Test đơn vị dựng dict tay theo artifact nên PASS — không bắt được.

**Sửa:** `face_module_v7.process_frame` phát thêm alias generic theo đúng
thứ tự canonical 52 blendshape như train (đủ 20 cặp mới phát) + 3 key pose.
Model + artifact giữ nguyên (đóng băng NK-12). Sau sửa: ML chấm 118/118,
hướng đúng (Stroke median 77.8 vs NonStroke 6.8). Hồi quy 10/10 + 58/58 +
8/8 PASS.

**Bài học:** (1) đường nối ML phải smoke bằng dữ liệu thật qua đúng đường
khởi tạo production; (2) tên key artifact ↔ runtime cần test đối chiếu
2 chiều; (3) fallback im lặng phải ghi log ít nhất 1 lần đầu xảy ra.
