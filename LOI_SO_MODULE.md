# 📓 LỖI SỔ MODULE — NHẬT KÝ NGHIÊN CỨU PSCS
## File ghi lỗi tổng hợp TẤT CẢ 5 MODULE

> **Cách dùng:** Mỗi khi phát hiện lỗi ở module nào → ghi vào section của module đó.
> Copy các entry có dấu 📌 sang **sổ nhật ký nghiên cứu**.
> Quy ước trạng thái: 🔴 ĐANG LỖI | 🟡 CẦN KIỂM TRA | ✅ ĐÃ FIX

**Ngày lập:** 03/09/2026
**Quyết định kèm theo:** Module 5 chuyển từ Visual Field sang **Radar LD2450** (theo TONG_QUAN_DU_AN_FINAL.md v6.0)

---

## 📖 MỤC LỤC
- [Module 1 — Face](#-module-1--face-asymmetry)
- [Module 2 — Speech](#-module-2--speech-dysarthria)
- [Module 3 — Arm](#-module-3--arm-weakness)
- [Module 4 — Gait](#-module-4--gait-abnormality)
- [Module 5 — Radar (mới)](#-module-5--radar-fall-detector-mới)
- [Lỗi hệ thống chung](#-lỗi-hệ-thống-chung)
- [Mẫu ghi lỗi mới](#-mẫu-ghi-lỗi-mới)

---

# 🔵 MODULE 1 — FACE ASYMMETRY

### M1-01: Kiến trúc face module phân mảnh (v1 / v6 / v7) 🟡
- **Ngày phát hiện:** 31/08 (DEBUG_MODULE_1_FACE.md)
- **File:** `src/detection/face_module.py`, `face_module_v6.py`, `face_module_v7.py`
- **Mô tả:** 3 version cùng tồn tại. `face_module_v7.py` dùng Tasks API (`mp.Image`, `face_landmarker_v2.task`) — version mới nhất. Debug 31/08 báo lỗi `mp.Image` namespace ở version cũ.
- **Rủi ro:** Import nhầm file cũ → lỗi runtime khó đoán.
- **Việc cần làm:** Chốt dùng v7, đưa các file cũ vào `backups/` hoặc xóa.
- **Trạng thái:** 🟡 CẦN KIỂM TRA / DỌN DẸP

### M1-02: Threshold chưa có nguồn khoa học 🟡
- **Ngày ghi nhận:** 30/08 (THANG_DO_CHUAN_QUOC_TE.md)
- **Chi tiết:** Mouth asymmetry 25%, Eye deviation 30°, Face tilt 10°, Nasolabial fold, Forehead — **đều chưa có DOI/paper**
- **Việc cần làm:** Search PubMed/Google Scholar (kế hoạch Ngày 2 của lịch, chưa xong)
- **Trạng thái:** 🔴 ĐANG LỖI (chưa có nguồn)

---

# 🟢 MODULE 2 — SPEECH DYSARTHIRIA

> Chi tiết 10 lỗi đã fix: xem `LOI_DA_GAP_MODULE2.md`

### M2-01: Gọi sai method `analyze_speech` → CRASH ✅ (fix 03/09)
- `module2_main.py` gọi method chỉ tồn tại ở v1 → đổi thành `predict_dysarthria()`

### M2-02: ML model load fail do hard-code kiến trúc ✅ (fix 03/09)
- Kiến trúc `[256,128,64]` hard-code không khớp checkpoint `[128,64,32]` → giờ **auto-detect từ checkpoint**

### M2-03: ML model dùng random weights khi load lỗi ✅ (fix 03/09)
- **Bài học quan trọng:** gán `self.ml_model = DysarthriaClassifier()` TRƯỚC `load_state_dict()` → khi fail vẫn giữ model random → predict sai không báo lỗi. Fix: `ml_model = None` trong except (fail-safe về rule-based).

### M2-04: 61 debug-marker làm hỏng chuỗi ✅ (fix 03/09)
- `print("= - module2_main.py:44"*70)`, `Normal: 100180` → công cụ debug ghi đè vào string literal

### M2-05: Unicode tiếng Việt lỗi trên console Windows ✅ (fix 03/09)
- cp1252 không in được tiếng Việt → thêm `sys.stdout.reconfigure(encoding='utf-8')`

### M2-06: Thiếu VAD (lọc nhiễu) ✅ (fix 04/09)
- Lịch yêu cầu webrtcvad — **quyết định dùng energy-based VAD tự viết (numpy)** thay vì webrtcvad vì: webrtcvad cần C compiler, dễ fail trên Windows/Python 3.13, và năng lượng khung đủ cho mục đích lọc nhiễu nền.
- Method mới: `detect_voice_activity()` → speech_ratio, noise_floor, has_speech
- **Lưu ý thiết kế:** VAD chỉ dùng cho metrics + phát hiện NO_SPEECH, KHÔNG cắt audio đưa vào ML (model train trên audio nguyên bản — cắt sẽ lệch phân phối features)
- **Trạng thái:** ✅ ĐÃ FIX (test: silence → ratio 0.0; voice → ratio 1.0)

### M2-07: WPM chưa so với baseline cá nhân ✅ (fix 04/09)
- Đã thêm: `set_baseline()`, `save_baseline()`, `load_baseline()`, `calibrate_baseline(30s)`
- `module2_main.py` có mode mới `--mode calibrate`, tự load `data/baselines/user_default.json` nếu có
- Rule-based score giờ so WPM với baseline (lệch >40% = bất thường) thay vì ngưỡng cứng 100-180
- **Trạng thái:** ✅ ĐÃ FIX

### M2-08: Test 10 normal + 5 dysarthria thực tế ✅ (đã chạy 04/09, TORGO thật)
- **Test script:** `module2/test_module2_full.py` (--extended để chạy full)
- **Kết quả lần 1 (đúng lịch: 10 normal + 5 dys):**
  - TPR **100%** ✅ (5/5 dys phát hiện, 4/5 ở DANGER >87%)
  - FPR **10%** ❌ (1/10 normal — file MC04S01 prob 33.9%, vượt ngưỡng 30% rất ít)
  - Accuracy 93.3% | Avg prob: NORMAL 8.5% vs DYS 85.0%
- **Kết quả extended (28 normal + 27 dys = 55 session, 1 file/session):**
  - TPR **85.2%** | FPR **3.6%** ✅ | Accuracy 90.9%
  - ROC Youden's J = 1.00 tại ngưỡng **51%** → FPR 0.0%, TPR giữ 85.2%
- **Số liệu xuất file:** `test_results/module2_test_20260904_*.json/.csv`
- **Trạng thái:** ✅ ĐÃ TEST — 2 mục còn mở: M2-10, M2-11

### M2-09: Model ML cũ `speech_classifier_20260828_175927.pth` (100% synthetic) 🟡
- Model này overfit trên synthetic data (accuracy 100% phi thực tế). Đã chuyển module2_main sang `speech_torgo_20260828_211130.pth` (83.07% TORGO — dữ liệu thật).
- **Việc cần làm:** Không dùng model synthetic cho báo cáo; cân nhắc xóa/archived.
- **Trạng thái:** 🟡 ĐÃ XỬ LÝ MỘT PHẦN

### M2-10: 4/27 dys bị miss trong extended test — window headMic gần như im lặng ✅ (fix 06/09)
- **Ngày phát hiện:** 04/09 | **Ngày fix:** 06/09
- **Nguyên nhân gốc:** session TORGO có hàng trăm wavs prompt ngắn; file `_0001` của headMic thường là calibration IM LẶNG (speech_ratio 0.05–0.08) — `first_wav()` chọn nhầm file xấu. Model train trên câu đọc CÓ TIẾNG → audio im lặng pad zeros làm features lệch.
- **Cách fix (2 tầng, trong `test_module2_full.py` + method mới `select_best_window`/`extract_window` trong `speech_module_v2.py`):**
  1. Tầng 1: quét tối đa 5 file của session, chọn file có speech_ratio cao nhất (VAD)
  2. Tầng 2: trong file đó, chọn cửa sổ 5s có speech_ratio cao nhất (quét cumsum năng lượng khung, hop 0.5s)
  3. **v2 bổ sung:** predict 3 cửa sổ tốt nhất → lấy MEDIAN prob (xem M2-12 vì sao)
- **Kết quả A/B (extended 55 session):** first5 cũ: TPR 85.2%/FPR 3.6% → **median3: TPR 96.3%/FPR 14.3% @th30; TPR 96.3%/**FPR 0.0%** @th56 (Youden J = 1.00)**
- **Bài học:** (1) chọn `first_wav` = mẫu ngẫu nhiên xấu; (2) median 3 cửa sổ = consensus chống outlier đơn; (3) chỉ 1 miss còn lại (F01 headMic — có 1 window 99.4% nhưng 2 window thấp, median đánh mất — trade-off chấp nhận)
- **Còn nợ:** median3 consensus mới có trong test script; cần đưa vào `module2_main.py` production (ghi 15s → 3 cửa sổ)
- **Trạng thái:** ✅ ĐÃ FIX (test script) — production integration TODO

### M2-12: Regression FPR khi dùng best-window ĐƠN (32.1%) ✅ (fix 06/09)
- **Ngày phát hiện:** 06/09 (lần thử đầu của M2-10)
- **File:** `module2/test_module2_full.py` (bản v1 của fix M2-10)
- **Mô tả:** Chọn duy nhất 1 cửa sổ "có tiếng nhất" → TPR tăng 88.9% nhưng **FPR tăng vọt 3.6% → 32.1%** (9/28 normal báo nhầm). Chọn cửa sổ TO NHẤT/RÕ NHẤT áp cho CẢ 2 nhóm → giọng normal cũng được phân tích ở chế độ "nói to rõ", model MLP 48 features nhạy → phân phối lệch lên.
- **Cách fix:** phân tích 3 cửa sổ tốt nhất, prob cuối = **MEDIAN** (kết quả: FPR 14.3% @th30, 0.0% @th56). 4 FP còn lại đều có phương sai cao giữa windows (vd 55.1/19.2/69.4) — model bất định với giọng biên.
- **Bài học:** extreme-of-selection (chọn max) amplifies noise; consensus (median) là cách đúng khi input không đồng nhất. Luôn A/B test khi thay đổi tiền xử lý.
- **Trạng thái:** ✅ ĐÃ FIX

### M2-11: Quyết định ngưỡng báo động 30% hay 51%/56% 🟡 (số liệu mới 06/09)
- ROC extended (median3, 06/09): ngưỡng 30% → TPR 96.3%/**FPR 14.3% FAIL**; ngưỡng **56% → TPR 96.3%/FPR 0.0% PASS cả 2 target** (Youden J = 1.00)
- **Khuyến nghị mới:** ALARM = 56% (median3); giữ 30% chỉ làm tín hiệu MONITOR sớm đưa vào Fusion (không tự báo động)
- **Trạng thái:** 🟡 CẦN QUYẾT ĐỊNH CUỐI (áp vào module + module2_main sau khi có Defense Layer 3)

---

# 🟠 MODULE 3 — ARM WEAKNESS

### M3-01: torch.load `weights_only` không tương thích PyTorch cũ ✅
- Đã có version check trong `arm_module.py` (line 102-107) — xác nhận 03/09

### M3-02: NIHSS mapping thiếu score 4 ✅
- Đã trả đủ 0-4 trong `_map_to_nihss` (line 381-401) — xác nhận 03/09

### M3-03: Threshold chưa có nguồn khoa học 🔴
- Arm drop 100px (pixel, không phải đơn vị thật!), Movement range 20°, Asymmetry 25°, Speed ratio 0.5 — **chưa có DOI/paper**
- **Lưu ý thêm:** đo bằng pixel là điểm yếu khi phỏng vấn (khoảng cách camera thay đổi → px thay đổi)
- **Trạng thái:** 🔴 ĐANG LỖI

---

# 🟣 MODULE 4 — GAIT ABNORMALITY

### M4-01: YOLO keypoints dùng nhầm `.xyxy` ✅
- Đã sửa dùng `.xy` (gait_module.py line 363-364, có comment) — xác nhận 03/09

### M4-02: Stride length tính sai (time thay vì value diff) ✅ (cần verify test)
- Code hiện tại dùng `np.abs(np.diff(value))` (line 126) — khớp gợi ý fix. Cần chạy test thực tế xác nhận.

### M4-03: Khủng hoảng thống kê (n=10, FPR 33.3%) ✅ (fix 30/08)
- ROC optimization threshold 0.30→0.35, FPR 32.8%→11.3%, accuracy 86.13%
- Xem `NGAY1_RESULT_MODULE4_FIX.md`, `roc_analysis_module4.py`

### M4-04: Symmetry threshold chưa có nguồn 🔴
- Stride/Cadence/Variability ✅ có nguồn (Hausdorff 2005, Menz 2003, Hausdorff 2007); **Symmetry <15% chưa có**
- **Trạng thái:** 🔴 ĐANG LỖI (1/4 threshold)

---

# 🔴 MODULE 5 — RADAR FALL DETECTION (MỚI)

> **Quyết định 03/09:** Chuyển Module 5 từ Visual Field sang Radar LD2450 theo TONG_QUAN_DU_AN_FINAL.md.
> Code cũ `visual_module.py` + model `visual_field_20260831_193418.pth` → chuyển thành legacy/archive.

### M5-01: Chưa có code radar nào 🔴 (module mới)
- **Cần build theo lịch Ngày 11-12 (12-13/09):**
  1. LD2450 + UART-USB: `serial.Serial('COM3', 115200)`
  2. `radar_module.py`: parse x, y, velocity + lọc nhiễu
  3. Fall detection proxy: position_change > 1m + inactivity > 45s
  4. Audio fusion AND-gate: `radar_fall AND audio_abnormal → ALERT`
- **Trạng thái:** 🔴 CHƯA BẮT ĐẦU

### M5-02: Phần cứng ✅ (đã mua 03/09)
- **ĐÃ MUA 1× LD2450** + UART-USB adapter
- **Quyết định thiết kế:** hệ thống chỉ dùng **1 radar duy nhất** (không phải 2 như TONG_QUAN v6.0). 1 radar đặt vị trí trung tâm (phòng khách). Code fall detection phải tối ưu cho 1 điểm đo duy nhất.
- **Trạng thái:** ✅ ĐÃ MUA — cập nhật kiến trúc: 1 radar, không còn radar phòng ngủ/phòng tách riêng

### M5-03: Module Visual cũ có vấn đề đã ghi nhận 🟡 (legacy)
- DEBUG_MODULE_5_VISUAL.md (31/08): 4 critical (MediaPipe API, random features, placeholder features, EAR landmarks)
- Đã xác nhận 03/09: random features đã bỏ (`visual_module.py` line 149, 167). Các issue còn lại không ưu tiên vì module chuyển sang radar.
- **Trạng thái:** 🟡 LEGACY — không đầu tư thêm

---

# ⚙️ LỖI HỆ THỐNG CHUNG

### SYS-01: Thiếu Fusion Engine ✅ (làm 04/09, sớm 1 ngày so với lịch Ngày 5)
- `src/fusion/fusion_engine.py` — class `FusionEngine`
- Thiết kế: adapter chuẩn hóa 5 module (mỗi module đặt tên prob khác nhau) → weighted average có renormalize (arm 0.30, face/speech 0.20, gait/radar 0.15) → luật FAST R1 (≥2/3 dấu hiệu prob≥50 → EMERGENCY) + R2 (1 module prob≥80 → WARNING) → NIHSS ước tính 4 items (max 15) → 4 risk levels
- Test 7/7 kịch bản PASS (`venv/Scripts/python.exe src/fusion/fusion_engine.py`)
- **Tài liệu phỏng vấn:** `file_logic_giai_thich_code_FusionEngine.md` (công thức, ví dụ tính tay, 10 Q&A)
- **Trạng thái:** ✅ HOÀN TẤT v1 — chờ tích hợp Dashboard (Ngày 5-6) + radar module (sẽ thêm key 'radar')

### SYS-02: Thiếu Alert System ✅ (làm 06/09, lịch là Ngày 5)
- `src/alerts/alert_system.py` — class `AlertSystem` + `src/alerts/__init__.py`
- Thiết kế: `process_fusion_result()` (auto từ FusionEngine) + `send_alert(score, message)` (manual, API lịch Ngày 5); quyết định mức: score>=80 hoặc risk EMERGENCY → EMERGENCY, score>=50/WARNING → WARNING, còn lại chỉ log INFO
- 3 kênh: buzzer `winsound` (pattern 880Hz / 1200-900Hz theo mức, guard import cho non-Windows), log JSONL `logs/alerts/alerts_YYYYMMDD.jsonl` (append-only crash-safe), Zalo stub (chỉ EMERGENCY, TODO access token OA)
- **Cooldown per-level 60s** chống spam (WARNING không chặn EMERGENCY); `suppressed_count` cho Dashboard; `get_stats()` cho Dashboard
- Test 7/7 PASS (5 mức + cooldown + manual): `venv/Scripts/python.exe src/alerts/alert_system.py`
- **Tài liệu phỏng vấn:** `file_logic_giai_thich_code_AlertSystem.md`
- **Trạng thái:** ✅ HOÀN TẤT v1 — Zalo thật + cooldown riêng theo mức = future

### SYS-08: Test cooldown Alert System tự mâu thuẫn ✅ (fix 06/09)
- **File:** `src/alerts/alert_system.py` (hàm `test_alert_system`)
- **Mô tả:** Lần 1 set `cooldown_seconds=0` cho cả test → kịch bản cooldown không bao giờ chặn (FAIL #6). Lần 2 bật 60s cho toàn bộ → kịch bản 5 và 7 (cùng mức EMERGENCY chạy liên tiếp <60s) bị cooldown chặn oan (FAIL #5, #7).
- **Nguyên nhân:** trộn 2 loại kịch bản (quyết định MỨC vs hành vi COOLDOWN) vào 1 instance với 1 config.
- **Cách fix:** chia test 3 nhóm — Nhóm 1 mức (cooldown=0), Nhóm 2 cooldown (bật 60s), Nhóm 3 manual (tắt lại 0).
- **Bài học:** test hành vi thời gian (cooldown/rate-limit) phải tách scope; 1 instance = 1 config = 1 nhóm assert.
- **Trạng thái:** ✅ ĐÃ FIX (7/7 PASS)

### SYS-03: Thiếu 4-Layer Defense Engine 🔴
- `src/defense/` RỖNG. Lịch ngày 12: Layer 1 Calibration (15 phút) → Layer 2 Context Awareness → Layer 3 Temporal Analysis → Layer 4 Adaptive Threshold. Target FPR <5%
- **Deadline theo lịch:** 12-13/09

### SYS-04: Thiếu NIHSS Estimator + Triage + Handoff 🔴
- Chỉ có NIHSS mapping rời rạc trong từng module. Chưa có: `nihss_estimator.py` (target r≥0.85), `triage_engine.py` (subtype/severity/hospital), `handoff_system.py` (report + QR + PDF)
- **Deadline theo lịch:** Tuần 3 (16-22/09)

### SYS-05: Dashboard Flask, kế hoạch là Streamlit 🟡
- `webapp/app.py` (Flask) tồn tại; lịch ghi Streamlit. Cần quyết định giữ Flask hay chuyển.
- **Trạng thái:** 🟡 CẦN QUYẾT ĐỊNH

### SYS-06: Bảng cơ sở khoa học Excel chưa tạo 🔴
- Lịch Ngày 2 yêu cầu `BANG_CO_SO_KHOA_HOC.xlsx` (5 sheets) — hiện chỉ có file MD
- **Trạng thái:** 🔴 CHƯA LÀM

### SYS-07: 0 thư xác nhận bác sĩ / 0 validation lâm sàng 🔴
- Đã ghi trong DANH_GIA_SAN_PHAM_Y_TE.md — kế hoạch Tuần 3-4
- **Trạng thái:** 🔴 CHƯA LÀM

---

# ✍️ MẪU GHI LỖI MỚI

```
### M<x>-<số>: <Tên lỗi ngắn> 🔴|🟡|✅
- **Ngày phát hiện:** DD/MM
- **File:** <đường dẫn>
- **Error message:**
```
<văn bản lỗi gốc>
```
- **Nguyên nhân:** <tại sao>
- **Cách fix:** <đã/làm gì> (kèm code BEFORE/AFTER nếu quan trọng)
- **Bài học:** <rút ra cho nhật ký>
- **Trạng thái:** 🔴/🟡/✅
```

---

**Cập nhật lần cuối:** 06/09/2026 — M2-10 fix (median3 consensus: TPR 96.3%, ROC th56 FPR 0%) + Alert System v1 (SYS-02 ✅)
**Số lỗi đang mở:** M1: 2 | M2: 2 | M3: 1 | M4: 1 | M5: 1 | SYS: 5 = **12 mục cần làm**
**Kết quả Module 2 cuối cùng:** Extended 55 session, median3 consensus — TPR 96.3% (26/27), FPR 14.3% @th30 / **FPR 0.0% @th56 (Youden J=1.00)**, Accuracy 90.9%
