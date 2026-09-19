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

### M5-01: Chưa có code radar nào ✅ (code 06/09, sớm so với lịch 12-13/09) — chờ cắm hardware
- `src/detection/radar_module.py` — `LD2450Parser` (buffer trượt, frame 30 bytes `AA FF 03 00` + 3×8 bytes int16 LE + tail `55 CC`, chống frame giả) + `RadarModule`
- Fall proxy theo spec: |Δx|+|Δy| > 1m trong <2s (+40); bất hoạt >45s sau biến động (+60); audio AND-gate ×0.3 khi audio bình thường; fallback SIMULATION 3 kịch bản (normal/fall/wander) cho demo không cắm radar
- Output format FusionEngine: `{'fall_prob', 'status', 'nihss_score': 0, 'metrics'}`
- Test trong `src/test_all_metrics.py` G1-G6 (parser + gate + format) PASS
- **Trạng thái:** ✅ CODE + SIM DONE — 🔴 chờ cắm LD2450 thật test UART (baud 256000, không phải 115200 như lịch ghi)

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

### SYS-03: Thiếu 4-Layer Defense Engine ✅ (code 06/09, sớm so với lịch 12-13/09)
- `src/defense/defense_engine.py` — class `DefenseEngine`
- L1 `calibrate()` baseline cá nhân; L2 `update_context()` phân loại EXERCISE/WALKING/TALKING/REST → CONTEXT_SUPPRESS (đánh dấu defense_note, KHÔNG mất prob — suppress hoàn toàn có thể che đột quỵ thật khi đang vận động); L3 `verdict()` persistent >30s + ≥60% mẫu ≥50 mới alert (warming-up cho qua — fail-safe), trend 5p; L4 adaptive floor vùng xám [30,40) −5 điểm chỉ khi đã calibrate
- **Trạng thái:** ✅ DONE (test F1-F6 PASS) — chờ tích hợp dài hạn với baseline thật 15 phút

### SYS-04: Thiếu NIHSS Estimator + Triage + Handoff ✅ (code 06/09, sớm so với lịch Tuần 3)
- `src/fusion/nihss_estimator.py`: band prob→item ĐÚNG thang Brott 1989 (item4 ≤3, item5/6 ≤4, **item10 ≤2 → subtotal max 13 KHÔNG phải 15**); CI Monte Carlo ±10 prob, 1000 lần, margin 1.96·SD
- `src/fusion/triage_engine.py`: subtype hint rule-based (base 80/20, +30 đau đầu, +20 nôn, −20 từ từ, −15 speech-dominant, clamp 5-95); severity MILD<6/MODERATE 6-10/SEVERE≥11 + nâng bậc WORSENING; hospital từ CSV
- `src/handoff/handoff_system.py`: timeline T0-T4 + report JSON 8 khối + QR tóm tắt + PDF reportlab font Arial (có dấu TV)
- `data/hospital_database.csv`: 6 BV/Phòng khám TP.HCM
- **Trạng thái:** ✅ DONE (test H1-H4, I1-I4, J1-J3 PASS)

### SYS-05: Dashboard Flask, kế hoạch là Streamlit ✅ (quyết định + làm 06/09)
- **Chốt: Streamlit** — `app_family.py` (app gia đình, giám sát liên tục 3s/fragment, 5 tab: Giám sát/Báo động feed/Kết quả NIHSS/Bệnh viện Handoff/Kiểm tra nói)
- Zalo → APP IN-FEED (người thân mở app thấy tin nhắn báo động ngay)
- `webapp/` Flask → legacy, không phát triển thêm
- **Trạng thái:** ✅ DONE (HTTP 200 verified)

### SYS-06: Bảng cơ sở khoa học Excel chưa tạo ✅ (làm 06/09)
- `BANG_CO_SO_KHOA_HOC.xlsx` — 5 sheets, 33 dòng (generate bằng `scripts/generate_science_table.py`)
- **Trung thực về trạng thái nguồn:** VERIFIED (có DOI/PMID) / CANONICAL (kinh điển — cần check PubMed trước khi in poster, search quota hết đến 13/09) / SELF-DESIGN (tự thiết kế — cần validate lâm sàng)
- **Trạng thái:** ✅ DONE — nợ verify DOI online sau 13/09

### SYS-07: 0 thư xác nhận bác sĩ / 0 validation lâm sàng 🟡 (đã chuẩn bị hồ sơ 06/09)
- Tool: `src/fusion/validation_metrics.py` (pearson/MAE/CM/ROC/Youden/sign-test + evaluate_nihss_study/detection_study)
- Quy trình: `docs/kiem_dinh_y_khoa/QUY_TRINH_KIEM_DINH_Y_KHOA.md` (Phần A 50 video NIHSS, B 100 kịch bản, C 2 thư)
- Mẫu thư: `docs/kiem_dinh_y_khoa/MAU_THU_XAC_NHAN_BAC_SI.md` (Mẫu 1 thư xem xét consultation + Mẫu 2 phiếu đồng chấm NIHSS + checklist đi gặp)
- **Trạng thái:** 🟡 HỒ SƠ SẴN SÀNG — 🔴 chưa đi gặp bác sĩ, chưa chạy 50 video (Tuần 3-4)

### SYS-09: NameError `sys` không định nghĩa ×3 ✅ (fix 06/09)
- **File:** `src/defense/defense_engine.py`, `src/fusion/nihss_estimator.py`, `src/fusion/triage_engine.py`
- **Error:** `NameError: name 'sys' is not defined` khi chạy `__main__`
- **Nguyên nhân:** dùng `sys.stdout.reconfigure(...)` trong `if __name__ == '__main__'` nhưng quên `import sys` ở đầu file (copy pattern từ file khác không kèm import)
- **Cách fix:** thêm `import sys` vào cả 3 file
- **Bài học:** pattern `sys.stdout.reconfigure` dùng ở `__main__` — file nào có block này phải import sys ở đầu; lỗi chỉ lộ khi chạy trực tiếp file, import từ module khác thì KHÔNG lộ → dễ sót. Test suite chạy trực tiếp từng file mới bắt được.
- **Trạng thái:** ✅ ĐÃ FIX

### SYS-10: KeyError 'VN-Bold' khi xuất PDF handoff ✅ (fix 06/09)
- **File:** `src/handoff/handoff_system.py`
- **Error:** `KeyError: 'VN-Bold'` (reportlab font registry)
- **Nguyên nhân:** chỉ đăng ký `Arial` thành 'VN' nhưng code dùng `font + '-Bold'` cho tiêu đề → lookup fail
- **Cách fix:** đăng ký thêm `arialbd.ttf` thành 'VN-Bold' (và italic nếu cần)
- **Bài học:** reportlab phải đăng ký TỪNG variant font (regular/bold/italic) riêng — không tự suy từ font cha
- **Trạng thái:** ✅ ĐÃ FIX

### SYS-11: `distance_km` rỗng trong hospital_database.csv → gợi ý BV sai ✅ (fix 06/09)
- **File:** `data/hospital_database.csv` + `src/fusion/triage_engine.py`
- **Mô tả:** dòng "Phòng khám" có `distance_km` rỗng → Mild (chọn gần nhất) nên gợi ý BV Nhân dân 115 (4.5km) thay vì phòng khám gần nhất; sort bị lỗi giá trị 999
- **Cách fix:** điền distance_km thật (1.2); `_b()` đọc an toàn + guard `float(h.get('distance_km') or 999)`
- **Bài học:** dữ liệu CSV đầu vào phải validate (cột số không được rỗng); test I4 có assert "CSV sạch"
- **Trạng thái:** ✅ ĐÃ FIX

### SYS-12: 4 test FAIL trong lần chạy đầu test suite mở rộng (đều là lỗi THIẾT KẾ TEST) ✅ (fix 06/09)
- **File:** `src/test_all_metrics.py` (section G, K)
- **Chi tiết:**
  1. K2: thiếu `f1_score` trong import list → NameError
  2. K6: assert `threshold > 25` quá chặt — Youden chọn đúng th=25.0 (biên cụm normal linspace 5-25, 1 FP → Spec 98% vẫn PASS target)
  3. G4/G5: kịch bản sim 'fall' PHA-DEPENDENT (random theo giây) → prob 18/40/100 tùy pha; lần đầu kỳ vọng cứng 30
- **Cách fix:** import đủ; nới assert biên (>=25); **monkeypatch `rm.read_targets`** trả vị trí CỐ ĐỊNH (nằm yên 150,60) → deterministic: +60 bất hoạt, gate ×0.3 → 18 (audio normal) / 60 DANGER (audio bất thường)
- **Bài học:** (1) test phải deterministic — không phụ thuộc random/time; mock input thay vì kỳ vọng may rơi đúng pha; (2) assert biên ngưỡng tối ưu phải theo logic Youden (có thể chọn ngay mép dữ liệu); (3) 58/58 PASS sau fix — code sản phẩm KHÔNG đổi gì
- **Trạng thái:** ✅ ĐÃ FIX

### SYS-13: Lỗi cú pháp f-string trong train_face_model_rtx3050.py ✅ (phát hiện 06/09 khi gom code KHKT)
- **File:** `src/training/train_face_model_rtx3050.py` line 528
- **Error:** `SyntaxError: invalid syntax` — `print(f   "   CUDA Version: ...")` (có khoảng trắng giữa `f` và dấu nháy)
- **Nguyên nhân:** cùng kiểu hỏng bởi debug tool như M2-04 (debug marker ghi đè chuỗi literal) — file train face đã chạy được trước đó nên nghi là hỏng sau lần sửa gần đây, không ai compile-check file train
- **Cách fix:** sửa `print(f"   CUDA Version...")`; phát hiện nhờ bước compile-check (py_compile) toàn bộ 35 file khi gom vào `Golden_Watch_KHKT/`
- **Bài học:** (1) script train chạy 1 lần rồi bỏ → không bao giờ được regression-check; compile-check phải là bước bắt buộc khi "chốt phiên bản"; (2) grep pattern `f[[:space:]]{2,}"` để săn hỏng cùng loại
- **Trạng thái:** ✅ ĐÃ FIX (cả bản gốc + bản KHKT)

---

# 📅 NHẬT KÝ 07/09 — LẤY DATASET THẬT TỪ fga_project + ĐÁNH GIÁ ĐỘC LẬP (Lỗi_ngày.md)

> Ngày 07/09 có đánh giá độc lập toàn đề tài → file `Lỗi_ngày.md` (L-01..L-16, D-01..D-13).
> Các mục dưới là việc đã LÀM theo danh sách đó.

### M4-05: Train gait trên window-random split → LEAKAGE subject ✅ (phát hiện + fix 07/09)
- **File:** `training/train_gait_model.py` (v1) → tạo `training/train_gait_model_v2.py`
- **Nguyên nhân:** v1 chia window NGẪU NHIÊN train/test → window của cùng 1 người nằm cả 2 tập → accuracy 86.13% là số phình; không giám khảo nào chấp nhận
- **Cách fix:** v2 dùng **LOSO (Leave-One-Subject-Out) 15 folds** trên dữ liệu THẬT PhysioNet Gait in Aging (15 subject: 5 già khỏe, 5 trẻ khỏe, 5 Parkinson; 162 windows)
- **Kết quả LOSO (window-level):** Sens 88.24% CI95 [65.66, 96.71] | Spec 88.28% CI95 [82.03, 92.55] | Acc 88.27% | AUC 0.884 | **Subject-level 14/15 = 93.33%** (miss o2 — người già 74 tuổi đi giống Parkinson)
- **Model mới:** `models/gait_classifier_v2_20260907_210433.pth` + scaler + metadata JSON (kèm citation nguồn + limitation rõ: Parkinson là proxy bất thường dáng đi, KHÔNG phải đột quỵ; thảm lực ≠ video pose)
- **Việc còn treo (M4-06):** ngưỡng runtime gait vẫn 64; ngưỡng Youden LOSO mới = 10 (×100). Quyết định cập nhật `gait_module.py` + ghi nhận đây là thay đổi threshold lần 2
- **Bài học:** số accuracy không ghi rõ đơn vị đánh giá (window? người?) là số VÔ NGHĨA; subject-level mới là đơn vị đúng
- **Trạng thái:** ✅ ĐÃ FIX (chờ quyết định threshold runtime)

### SYS-14: Golden_Watch_KHKT thiếu file .task thật → face demo chạy FALLBACK không nhận diện 🟡 (phát hiện 07/09)
- **Nguyên nhân:** KHKT/models chỉ có stub 9KB; file thật 3.75MB ở fga_project. FaceAsymmetryDetector rơi nhánh "numpy-based mode (no detection)" → score luôn 0 → fusion bỏ module face (INVALID NO_DETECTOR) mà không báo động lớn
- **Cách fix:** copy `face_landmarker_v2_with_blendshapes.task` + `face_landmarker.task` (3.75MB) sang KHKT. Evaluator chạy được ngay
- **Bài học:** (1) model asset phải nằm trong repo (hoặc download script + checksum); (2) cần 1 unit test "detector phải trả khác NO_DETECTOR trên 1 ảnh mặt thật" để chặn hỏng loại này khi chốt phiên bản; (3) đây là bằng chứng lỗi "demo chạy đẹp nhưng module câm lặng" — thêm vào checklist demo trước thi
- **Trạng thái:** 🟡 ĐÃ FIX file, chờ thêm test chặn regression

### M1-03: Chia tập face theo ẢNH ngẫu nhiên → leakage frame liên tiếp ✅ (fix code 07/09, chờ retrain GPU)
- **File:** `training/train_face_model.py` (đã vá split)
- **Nguyên nhân:** img_NNNN là frame trích từ video; chia ảnh ngẫu nhiên → frame gần-trùng nhau cả 2 tập → 93.75% là số phình
- **Cách fix:** block-aware split (block = dải img_NNNN // 50, kèm nhãn). Kiểm tra: 151 blocks, test 776 ảnh (249 stroke), **0 ảnh trùng giữa các tập**
- **Kèm theo:** `training/face_dataset_info.json` — xác nhận nguồn là dataset CÔNG KAI (sửa sai "tự thu" trong BANG_CO_SO sheet 2); cần bổ sung URL Kaggle chính xác sau 13/09 (quota reset)
- **Việc còn treo:** chạy lại train trên RTX3050 → thay số 93.75% bằng số block-split mới trong TẤT CẢ tài liệu
- **Trạng thái:** ✅ code, 🟡 chờ retrain

### M1-05: 🔴🔴 NHÃN ĐẢO NGƯỢC — model face 93.75% train trên nhãn SAI (phát hiện 07/09)
- **File:** `training/train_face_model.py` (prepare_dataset) + dataset "Annotated stroke and non stroke Dataset"
- **Bằng chứng:** kiểm tra toàn bộ file .txt: thư mục **NonStroke: 100% token đầu = "1"**; thư mục **Stroke: 100% token đầu = "0"** — token trong annotation ĐẢO NGƯỢC tên thư mục. Script cũ đọc `label = int(parts[0])` (giả định Stroke=1) → nhãn train bị ĐẢO HOÀN TOÀN so với thư mục
- **Hệ quả:** model cũ 93.75% học "đāu hướng ngược": nhãn 1 ("stroke") thực chất là ảnh NonStroke. Nếu wire model này vào runtime sẽ BÁO NGƯỢC. May mắn: runtime đang là rules (không dùng model này) nên sản phẩm demo chưa bị ảnh hưởng
- **Cách fix:** label lấy theo TÊN THƯ MỤC (NonStroke=0, Stroke=1), bỏ hẳn token .txt; bbox vẫn lấy từ .txt (parts[1:5]). Đã chạy lại train với: (a) nhãn thư mục, (b) block-split chống leakage, (c) mediapipe Tasks API
- **Bài học:** (1) KHÔNG BAO GIỜ tin nhãn trong file metadata khi có cấu trúc thư mục — luôn sanity-check token vs folder trên vài mẫu trước khi train; (2) "93.75%" cũ bỏ giá trị — mọi tài liệu phải thay bằng số mới; (3) phải có unit test "load 5 ảnh mẫu, assert nhãn khớp folder"
- **Trạng thái:** ✅ ĐÃ FIX CODE + retrain xong 07/09. **Kết quả trung thực trên test block-split (776 ảnh, nhãn đúng): Acc 67.9% | AUC 0.555 | Sens 2.4% (6/249) | Spec 98.9%** — model sụp về dự đoán đa số lớp (527 nonstroke vs 249 stroke, không class-weight). Kết luận: dataset internet này gần như KHÔNG mang tín hiệu phân biệt liệt mặt ở mức landmark; số "93.75%" cũ = nhãn đảo + leakage, vô giá trị và đã loại bỏ hoàn toàn
- **Quyết định (chốt 07/09):** (1) KHÔNG wire model PyTorch vào runtime; runtime vẫn là rules (AUC 0.638 — tạm tốt hơn model); (2) bằng chứng face chính sẽ đến từ **protocol B: data webcam thật thu nội bộ** (người khỏe + mô phỏng lép mặt có kiểm soát) — đúng domain triển khai; (3) dataset công khai chỉ dùng làm benchmark "khắc nghiệt" + ghi limitation trung thực trong báo cáo; (4) nếu muốn cứu model: class-weight + chọn subset landmark đối xứng thay vì 936 tọa độ thô — để sau protocol B
- **Kèm lỗi nhỏ fix cùng lúc (M1-05b):** mediapipe 1.0.x đã bỏ `solutions` API → FaceStrokeDataset chuyển sang Tasks API FaceLandmarker IMAGE mode (lấy 468 landmark đầu khớp input_size 936)

### M1-04: Đánh giá bộ RULES face (runtime thật) trên tập test block-split 🟡 (07/09)
- **File:** `training/evaluate_face_rules.py` (mới)
- **Lý do:** runtime face KHÔNG dùng model PyTorch 93.75% — runtime là rules (mouth 25%, eye 30%, tilt 10°, nasolabial 35%, forehead 30%). Bằng chứng phải đo trên cái ĐANG CHẠY
- **Smoke test 20 ảnh:** AUC 0.810, Youden th=54.1, Sens 57.1% (n=7), Spec 100% (n=6). ⚠️ Tỉ lệ NO_FACE cao (7/20 = 35%) — đã chẩn đoán: thử 3 cách crop (full/YOLO/corner) trên 80 ảnh → NO_FACE do **chất lượng dataset internet** (nhiều ảnh không frontal/mờ), KHÔNG phải lỗi crop (YOLO ≈ full image)
- **Kết quả FULL 776 ảnh test (`test_results/face_rules_eval_20260907_211506.json`):**
  - 478 ảnh có mặt hợp lệ, **298 NO_FACE (38.4%)** — giới hạn dataset, khi chạy webcam thật/live sẽ tốt hơn nhiều nhưng PHẢI nói rõ khi trình bày
  - AUC = **0.638** | Youden th=47.9 | Sens = **50.3%** CI [42.6, 58.0] | Spec = **73.2%** CI [68.1, 77.8]
  - → Khoảng cách LỚN so với 93.75% cũ (model PyTorch chưa wire + split leakage). Số này là baseline trung thực của bộ rules hiện tại
  - Hệ quả: face rules hiện KHÔNG đủ tốt để tự tin demo "93.75%". Việc bắt buộc: (1) lấy số retrain block-split (đang chạy); (2) protocol B webcam thật để đo môi trường thật; (3) cân nhắc wire model PyTorch vào runtime nếu nó thắng rules rõ rệt trên test block-split
- **Lưu ý phụ phát hiện:** comment trong code face ghi "dua tren Ross et al. 2021" nhưng BANG_CO_SO sheet 2 ghi SELF-DESIGN — mâu thuẫn nguồn (thêm vào D-01)
- **Trạng thái:** 🟡 đang chạy

### M1-06: 🔥 ĐỘT PHÁ — Dataset CÓ tín hiệu liệt mặt; AI học được khi dùng ĐÚNG đại diện đặc trưng (07/09)
- **Bối cảnh:** sau khi model 936 tọa độ thô thất bại (AUC 0.555), đặt câu hỏi khoa học: "dataset VÔ tín hiệu hay AI HỌC SAI CÁCH?" → thiết kế thí nghiệm 3 tầng (`training/test_face_signal.py`)
- **T1 — Cohen's d từng feature (stroke vs non, 478 ảnh test):**
  - mouth_ratio: 34.7 vs 18.9, **d = 0.759 (tín hiệu trung bình-mạnh)**
  - nasolabial_ratio: d = −0.561 | eye_ratio: d = −0.482 (hướng ÂM — dấu hiệu nhiễu nhãn trong dataset internet, ghi nhận trung thực)
  - face_tilt: d = 0.243 (yếu) | forehead_ratio: d = −0.072 (không tín hiệu)
- **T2/T3 — GroupKFold(5) THEO BLOCK (đánh giá ngoài fold, chống leakage):**
  - Logistic 5 features: **AUC 0.846** | MLP 5 features: **AUC 0.842**
  - Tham chiếu cùng dữ liệu: rules tuyến tính 0.638 | MLP 936 tọa độ thô 0.555
- **Kết luận khoa học:** thất bại trước đó do **ĐẠI DIỆN ĐẶC TRƯNG** (raw tọa độ + lệch lớp không xử lý), KHÔNG do dataset vô tín hiệu. 5 features bất đối xứng y khoa là đại diện đúng → AI + class-weight học được. Điều này giải thích vì sao "thêm ảnh" không cứu được model cũ và xác nhận hướng feature-engineering
- **Bài học:** (1) AI không phải phép màu — đại diện đặc trưng quyết định; (2) luôn tách "thí nghiệm chẩn đoán tín hiệu" (same-data OK) khỏi "số triển khai" (bắt buộc ngoài-fold); (3) hướng dính dấu trừ (nasolabial/eye) = cảnh báo nhiễu nhãn dataset → KHÔNG giải thích lâm sàng từ dấu này
- **Việc tiếp theo (M1-07):** train model nhỏ trên 5 features + class-weight, chọn ngưỡng Youden trên tập validation block, **chỉ wire vào runtime SAU khi protocol B (webcam thật) xác nhận cùng xu hướng** — không bao giờ công bố số này là "độ chính xác lâm sàng"
- **Trạng thái:** ✅ ĐỘT PHÁ đã ghi nhận, JSON: `test_results/face_signal_test_20260907_221456.json`

### SYS-15: 🔴 VẤN ĐỀ PHƯƠNG PHÁP — "tự thu dữ liệu = tự làm tự chứng minh?" (07/09, băn khoăn của nhóm)
- **Tình huống:** nhóm loay hoay: không được test bệnh nhân (cần chứng chỉ hành nghề), liên kết BV/chuyên gia vướng thủ tục → dự định tự thu dữ liệu trên người khỏe + mô phỏng thiếuнолет → BẮT xuất hiện nghi vấn "tự làm tự chứng minh, không có bệnh nhân thì đo độ chính xác kiểu gì?"
- **Phân tích (chốt cách trả lời chuẩn khoa học):**
  1. **Tách 2 loại validation:** (a) *analytical validation* = hệ có phát hiện đúng TÍN HIỆU thiết kế (lép mặt mô phỏng, nói líu mô phỏng, ngã thật, báo nhầm trên người khỏe) — protocol B trả lời được; (b) *clinical validation* = chính xác trên bệnh nhân thật — KHÔNG làm được, KHÔNG claim
  2. **Mỏ neo lâm sàng NGOÀI không cần bệnh nhân riêng:** 50 video NIHSS công khai có điểm bác sĩ (bệnh nhân thật, chấm bởi bác sĩ thật) → tương quan r/MAE là bằng chứng lâm sàng ngoài chân thực; TORGO + dataset công khai là mỏ neo ngoài thứ hai/thứ ba
  3. **Chống "tự chứng minh" bằng quy trình:** (a) protocol + tiêu chí thành công + ngưỡng VIẾT TRƯỚC khi thu (QUY_TRINH_KIEM_DINH_Y_KHOA.md ký ngày 06/09); (b) nhãn ground truth đến từ THIẾT KẾ kịch bản (nhóm FPR = chắc chắn khỏe) không do model tự đánh giá; (c) 2 người thực hiện — 1 diễn 1 ghi; (d) log TOÀN BỘ kịch bản kể cả fail, không chọn lọc; (e) hạn chế được ghi công khai
  4. **Câu chốt cho giám khảo:** "Chúng em KHÔNG và KHÔNG THỂ khẳng định độ chính xác lâm sàng vì chưa test bệnh nhân — đó là future work. Những gì sản phẩm chứng minh được hôm nay: phát hiện được tín hiệu thiếuнолет mô phỏng trong điều kiện nhà, không báo nhầm trên người khỏe, và tương quan với NIHSS do bác sĩ chấm trong video chuẩn. Định vị sản phẩm: công cụ sàng lọc hỗ trợ, không phải chẩn đoán."
- **Trạng thái:** ✅ đã chốt khung phương pháp; sẽ in kèm protocol B

### M5-01 (bổ sung): Script test radar hardware chuẩn bị sẵn cho 08/09 ✅
- **File:** `module5/test_radar_hardware.py` (mới)
- **Nội dung:** kiểm tra UART thô (nhắc baud 256000), 10 kịch bản có kỳ vọng (đi/ngồi/vẫy/không người/ngã đệm/ngã bất động 45s+/ngã im lặng gate audio/2 người), xuất JSONL từng sample + CSV tổng hợp `test_results/radar_hw/`
- **An toàn:** ngã chỉ xuống đệm, có người bảo hộ
- **Trạng thái:** ✅ sẵn sàng, chờ cắm hardware 08/09

---

# 📅 NHẬT KÝ 07/09 (BUỔI 3 — TÍCH HỢP 2 PHẦN + BỘ METRICS CHUẨN)

---

### M4-07: Model gait v2 KHÔNG nối vào đường camera (quyết định trung thực) ✅
- **Ngày phát hiện:** 07/09
- **File:** `src/detection/gait_module.py`, `app_family.py`
- **Nguyên nhân:** `GaitPoseDetector` (app) tính 6 chỉ số pose từ YOLO; model v2 học trên **8 đặc trưng thảm lực PhysioNet** (`extract_gait_features` cần time-series (N,2)) — 2 không gian đặc trưng KHÔNG tương thích. Nối model vào camera = bịa input = pseudo-science (giống bài học M1-05).
- **Cách fix:** camera gait giữ rule-based; model v2 chỉ dùng khi có cảm biến lực. Ghi chú trong header `app_family.py`.
- **Trạng thái:** ✅ ĐÓNG (quyết định kiến trúc có căn cứ)

### M1-07: Face ML 5 đặc trưng — TRAIN THẬT + artifact lưu `models/` ✅
- **Ngày:** 07/09. **Script:** `training/evaluate_all_metrics.py` (hàm `eval_face_ml`)
- **Kết quả (GroupKFold 5 theo block, class-weight, OOF):** Logistic AUC **0.845** [0.802–0.885], Acc 81.2%, Sens 72.6%, Spec 85.4%, F1 0.717 · MLP AUC 0.842 · Calibration plot + forest Cohen's d.
- **Artifact:** `models/face_asym_v2_5feat_20260907_223229.json` (coef + scaler + threshold 0.5)
- **Lưu ý:** CHƯA nối app — chờ protocol B (SYS-15b, 21–23/09); nếu đạt → flag `FACE_ML_ENABLED`.
- **Trạng thái:** 🟡 model sẵn sàng, chờ kiểm chứng vận hành

### SYS-16: Áp dụng bộ thang đo chuẩn AI + y tế (có trích bài báo) ✅
- **Ngày:** 07/09. **Tài liệu:** `BANG_THANG_DO_KIEM_DINH.md`
- **Nội dung:** toàn bộ số liệu module kèm Acc/Precision/Recall/Spec/NPV/F1/AUC + Wilson 95% CI (Wilson 1927) + bootstrap AUC CI + Youden + Silhouette (Rousseeuw 1987) + calibration (Hosmer–Lemeshow 1980); hồi quy NIHSS (MAE/RMSE/R²/Bland–Altman 1986/weighted-kappa Cohen 1968) ghi rõ **THIẾU** — chờ 50 video L-01A; chuẩn báo cáo TRIPOD+AI (BMJ 2024;385:e078378) + STARD 2015 (BMJ 2016;352:h5527).
- **Chạy:** `test_results/metrics_pack_20260907_223217/` — 6 dòng số liệu + 13 PNG; seed 42; 100% out-of-fold/LOSO.
- **Kết quả chính:** face_rules AUC 0.638 · face-ML 0.845 · gait LOSO 0.879 [0.742–0.961] (subject-15: 0.920) · speech 0.992 [0.968–1.0] · silhouette gait 2 lớp 0.681 / 3 nhóm 0.172 / face 0.049.
- **Trạng thái:** ✅ (cập nhật lại sau SYS-18/19)

### SYS-17: App 2 phần Camera/Radar + smoke test ✅
- **Ngày:** 07/09. **File:** `app_family.py` (v8.1)
- **Nội dung:** tách UI thành 📷 PHẦN 1 (giám sát camera + kiểm tra nói), 📡 PHẦN 2 (radar: chọn COM port thật qua `RadarModule.list_ports()` — bỏ cứng COM3, kịch bản SIM, lịch 120 nhịp + biểu đồ dịch chuyển/bất động), phần chung (Báo động/Kết quả NIHSS/Bệnh viện QR-PDF). Radar panel mới `tab_radar()`.
- **Kiểm chứng:** `py_compile` OK; suite `src/test_all_metrics.py` **58/58 PASS** sau sửa (không hồi quy).
- **Trạng thái:** ✅ (smoke 30 phút + LIVE radar chờ G1)

### SYS-18 (mới, kế hoạch): 50 video NIHSS → hoàn thiện metrics hồi quy 🟡
- MAE/RMSE/R²/Bland–Altman/weighted-κ máy-vs-chuẩn; khung 15–20/09 (KE_HOACH G2).
- **Trạng thái:** 🟡 chưa chạy

---

# 📅 NHẬT KÝ 08/09 — PROTOTYPE + BENCHMARK + 2 BUG

### M3-06: Arm crash khi có người trong khung (keypoints (17,2) ≠ (17,3)) ✅
- Benchmark phát hiện; fix ghép conf vào keypoints; chặn bởi SYS-14b test 3. Chi tiết L-24 (Lỗi_ngày.md).

### SYS-14b: Pre-flight regression `src/test_sys14b_regression.py` — 10/10 PASS ✅
- Chặn: stub .task (phát hiện `models/face_landmarker_v2.task` 8KB còn sót → đã thay file thật 3.67MB), arm crash, speech mất pitch (numpy), gait/radar load.
- Quy trình: chạy TRƯỚC mỗi buổi demo; exit 1 = không demo.

### SYS-20: Benchmark prototype RTX3050+C270+LD2450 ✅
- `module5/prototype_benchmark.py` → JSON + biểu đồ PNG. Chu kỳ 81ms/3000ms; C270 720p@32fps; CUDA bật (torch 2.14.0+cu126); numpy pin 2.3.4 (L-25).
- Số liệu điền vào PROTOTYPE_SPEC.md §5 + BAO_CAO_DU_AN.md §4.2.

### SYS-21: Đặc tả prototype + dàn ý báo cáo ✅
- `PROTOTYPE_SPEC.md` (BOM ≈1,01 triệu; lắp đặt; ảnh chụp cần có; rủi ro an toàn) — phục vụ Chế tạo 20đ.
- `BAO_CAO_DU_AN.md` — dàn ý 15 trang đúng cấu trúc KH 5.2.2, số thật điền sẵn, chỗ [ĐIỀN] ghi rõ.

### M1-07b: Face ML 5-feat nối app qua feature-flag (mặc định TẮT) ✅
- `src/detection/face_ml_5feat.py` load artifact JSON; sidebar bật/tắt; khi bật: score thay bằng ML prob, rules_prob giữ trong metrics để so. Smoke: ảnh stroke-typ 76.7% / non-typ 27.2%. Suite 58/58.

---

# ✍️ MẪU GHI LỖI MỚI

---

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

**Cập nhật lần cuối:** 08/09/2026 (buổi 5) — SYS-23 app v8.2 camera giám sát THỜI GIAN THỰC (thread camera 30fps + video fragment 7fps + khung xương YOLO từng khung + radar thread riêng) theo góp ý người dùng · L-28 face confidence 0.3 + HOLD 3s + HUD khung đo chính xác · AppTest 0 exception · 58/58 + 10/10 PASS.
**Cập nhật trước:** 08/09/2026 (buổi 4) — Prototype chốt RTX3050+C270+LD2450 (PROTOTYPE_SPEC) · benchmark 81ms/chu kỳ, C270 720p@32fps, CUDA torch cu126 (SYS-20) · bắt & sửa M3-06 arm keypoints crash + L-25 numpy pin 2.3.4 · SYS-14b pre-flight 10/10 (thay stub .task sót) · M1-07b face ML feature-flag OFF · SYS-22 bộ 5 biểu đồ định lượng (charts_accuracy_052303) · dàn ý báo cáo 15 trang (BAO_CAO_DU_AN.md) · 58/58 PASS.
**Cập nhật trước:** 07/09/2026 (buổi 3) — App tách 2 phần 📷Camera/📡Radar (SYS-17, bỏ COM3 cứng) ✅ · M4-07 quyết định không nối gait-PhysioNet vào camera ✅ · M1-07 face ML train thật AUC 0.845 + artifact ✅ · SYS-16 bộ thang đo chuẩn AI+y tế có trích dẫn (BANG_THANG_DO_KIEM_DINH.md) + metrics_pack 223217 ✅ · 58/58 PASS. Kế hoạch chủ: KE_HOACH_TICH_HOP_NGHIEM_THU.md · Nghiệm thu: KICH_BAN_NGHIEM_THU.md.
**Cập nhật trước:** 06/09/2026 (buổi 2) — Hoàn thành engine stack: SYS-03 Defense ✅, SYS-04 NIHSS+Triage+Handoff ✅, SYS-05 Streamlit app gia đình ✅, SYS-06 Bảng khoa học ✅, SYS-07 hồ sơ kiểm định sẵn sàng, M5-01 radar code+sim ✅, AlertSystem v2 (app in-feed) ✅, SYS-09..12 fix. Test suite mở rộng **58/58 PASS**. Push GitHub Golden-Watch commit `01d0bf5`.
**Số lỗi đang mở (cần làm tiếp):** M1: 2 (threshold nguồn, dọn version) | M2: 2 (M2-10 production integration, M2-11 chốt ngưỡng) | M3: 1 (nguồn threshold) | M4: 1 (nguồn symmetry) | M5: 1 (cắm LD2450 thật) | SYS: 1 (SYS-07 gặp bác sĩ + verify DOI sau 13/09) = **8 mục cần làm**
**Kết quả Module 2 cuối cùng:** Extended 55 session, median3 consensus — TPR 96.3% (26/27), FPR 14.3% @th30 / **FPR 0.0% @th56 (Youden J=1.00)**, Accuracy 90.9%
**Test suite tổng:** `src/test_all_metrics.py` — 58/58 PASS (A Speech 11, B Fusion 8, C Alert 4, D Models 4, E Artifacts 2, F Defense 6, G Radar 6, H NIHSS 4, I Triage 4, J Handoff 3, K Validation 6)

### SYS-22: Bộ 5 biểu đồ khoa học định lượng độ chính xác ✅
- **Script:** `training/plot_accuracy_charts.py` (seed 42, 100% OOF/LOSO)
- **Output:** `test_results/charts_accuracy_20260908_052303/` — A_ROC gộp 4 module · B_Precision-Recall (AP + tỷ lệ dương nền) · C_bar Acc/Prec/Sens/Spec/F1 ± Wilson CI tại ngưỡng vận hành · D_Sens-Spec theo ngưỡng + Youden (face-ML, gait) · E_forest AUC ± 95% CI bootstrap 1000 lần
- **Chú ý thống kê:** AUC gait LOSO dao động nhỏ giữa lần chạy (0.879–0.890) do khởi tạo ngẫu nhiên model — báo cáo ghi "~0.88±0.01" hoặc trích 1 lần + nói rõ seed; CI bootstrap [0.751–0.969] đã bao phủ cả 3 lần đo
- **Trạng thái:** ✅ chèn thẳng vào báo cáo/poster

### SYS-23: Refactor app → camera giám sát THỜI GIAN THỰC (góp ý người dùng) ✅
- **Ngày:** 08/09/2026 (buổi 5)
- **Góp ý gốc (người dùng tự tay chạy app):** "Khung phải đa dạng MediaPipe VÀ YOLO — là camera giám sát đặt trên cao quay thời gian thực; mở app phải quan sát chuyển động như camera giám sát thông thường, có bất thường thì báo/gửi tin nhắn trên app. Chụp — tính — load liên tục thì không hiệu quả."
- **Nguyên nhận:** v8.1 hiển thị 1 khung tĩnh/3 giây (kiểu slideshow) — không đáp ứng "camera giám sát"
- **Thiết kế mới (v8.2, 3 luồng song song):**
  - Thread `_camera_worker`: đọc webcam liên tục ~30fps → buffer có lock
  - `video_fragment` (0.15s ≈ 7 khung/giây): VIDEO TRỰC TIẾP + khung xương YOLO vẽ TỪNG KHUNG (≤3 người, `draw_live_pose`) + HUD MediaPipe (điểm méo mặt, ngưỡng 30, góc nghiêng)
  - `monitor_fragment` 2s: chỉ phân tích (face MediaPipe + arm/gait ML + Defense + Fusion + Alert → tin nhắn báo động trong app)
  - Thread `_radar_worker` riêng: radar quét 0.6s không chặn video; đổi radar = swap vào thread + đóng serial cũ
  - `yolo_lock`: 1 model YOLO dùng chung 2 thread
- **Kiểm chứng:** AppTest BẬT giám sát thật (chờ thread): 0 exception · frame 480×640 đổ liên tục · radar NORMAL · fusion 55 WARNING + alert bắn · 10/10 pre-flight + 58/58 unit test
- **Bài học:** app giám sát y tế phải theo mô hình "quan sát liên tục — phân tích định kỳ — báo động tức thì", không phải chụp-ảnh-tĩnh-chu-kỳ; tách thread đọc cảm biến khỏi thread phân tích là bắt buộc để video không giật
- **Hạn chế trung thực:** MediaPipe mặt vẫn cache 2s (mặt ít chuyển động — chấp nhận); nếu cần mượt từng khung → chạy MediaPipe trong video fragment (timestamp VIDEO mode phải tăng đơn điệu — cần xử lý thêm)
- **Trạng thái:** ✅

### SYS-24: Web server FastAPI + MJPEG thay Streamlit cho video giám sát ✅
- **Ngày:** 08/09/2026 (buổi 5) — theo góp ý người dùng: Streamlit re-run script mỗi tick làm video giật; đề xuất FastAPI local
- **File:** `web_server.py` — FastAPI + MJPEG (multipart/x-mixed-replace), đúng mô hình web camera giám sát/IP
- **Kiến trúc 3 thread:** camera 30fps (draw_live_pose YOLO từng khung + draw_face_hud MediaPipe) · radar worker riêng · analysis 2s (Defense→Fusion→Alert→NIHSS→Handoff)
- **Endpoints:** `/` dashboard CCTV offline (inline HTML/JS, không cần mạng) · `/video.mjpg` (~15–20fps, mở VLC được) · `/api/state` · `/api/radar` POST · `/patient` · `/alert/test` · `/handoff/qr` · `/handoff/pdf`
- **Bảo mật:** bind 127.0.0.1:5001 duy nhất — chỉ máy自身 truy cập, không lộ mạng
- **Số đo thật:** MJPEG 2.58MB/4s; QR 1103B; PDF 78KB; app_family.py (Streamlit) GIỮ NGUYÊN làm minh chứng nghiên cứu + tab kiểm tra nói
- **Bài học:** chọn framework theo bài toán — video real-time cần streaming server, không phải UI re-run; Streamlit phù hợp form/thí nghiệm (giữ cho speech test)
- **Trạng thái:** ✅

### SYS-25: Guards chống đoán sai + AGC mic xa + CHẾ ĐỘ HỌC 3 NGÀY (góp ý người dùng) ✅
- **Ngày:** 08/09/2026 (buổi 6) — ghi nhận đầy đủ trong Lỗi_ngày.md L-31
- **Thay đổi code:** face_module_v7 (laugh guard `_detect_expression` giảm 60% metric biểu cảm khi cười lớn đối xứng; PARTIAL_FACE khi bbox chạm mép khung — fusion/nihss thêm status missing) · speech_module_v2 (AGC peak<8%→x~4 khi mic xa) · web_server + app_family (vẽ 239 chấm landmark mặt trên khung) · **module mới `src/defense/personal_profile.py`**
- **Chế độ học 3 ngày:** khi lắp tại nhà, hệ thống quan sát + ghi median chỉ số (KHÔNG lưu ảnh) → đủ 3 ngày tự `calibrate()` baseline cá nhân → L4 Adaptive Floor hoạt động → giảm báo giả từ đặc điểm riêng (gù lưng, cong cột sống, nhech mép thói quen). Xóa file JSON = quét sạch dữ liệu cá nhân
- **Kiểm chứng:** laugh guard PASS · profile self-test PASS · 58/58 · 10/10 · web smoke OK (profile hiển thị trong /api/state + banner web)
- **Giới hạn trung thực:** dataset hiện có không có nhãn cười/gù/mép-quen — chưa train sâu được các ca này, phải thu thêm dữ liệu có nhãn (đề xuất mở rộng L-01A)
- **Trạng thái:** ✅

### SYS-26: Chuyển sang dùng MA TRẬN TƯ THẾ ĐẦU chính thức của MediaPipe Face Landmarker (góp ý người dùng) ✅
- **Ngày:** 08/09/2026 (buổi 6) — nguồn: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
- **CHUYỂN TỪ → SANG (tóm tắt 1 dòng):**
  - **TỪ:** tự tính nghiêng đầu THỦ CÔNG bằng 2 landmark 2D (điểm trán 10 → điểm cằm 152, atan2 so trục dọc) — chỉ đo được 1 góc nghiêng, dễ nhiễu khi đầu quay ngang
  - **SANG:** bật `output_facial_transformation_matrixes=True` trong `FaceLandmarkerOptions` — dùng MA TRẬN 4×4 chính thức mà FaceLandmarker trả về mỗi khung (ánh xạ mặt chuẩn → mặt thật), suy ra ĐỦ 3 góc Euler: **yaw (quay mặt trái/phải), pitch (ngả/cúi), roll (nghiêng vai)**
- **Sửa file `src/detection/face_module_v7.py`:**
  1. `FaceLandmarkerOptions` + `output_facial_transformation_matrixes=True`
  2. Hàm mới `_head_pose_from_matrix()` — trích yaw/pitch/roll từ khối quay R của ma trận theo công thức chuẩn trong docs MediaPipe
  3. `process_frame()` gắn `head_yaw/head_pitch/head_roll` (độ, làm tròn 1 chữ số) vào `raw_metrics` — CHỈ HIỂN THỊ, không đổi công thức điểm 3 metric cũ (score/NIHSS giữ nguyên để không phá kết quả đã nghiệm thu)
- **Sửa HUD (`web_server.py` + `app_family.py`):** thêm dòng `chuyen dau (matrix MP): quay X | nga Y do` ngay dưới dòng nghiêng đầu — người dùng thấy rõ gương mặt DỊCH CHUYỂN theo thời gian thực
- **Kiểm chứng (ma trận quay tổng hợp):** quay 30° quanh Y → yaw 30.0 ✓ · 35° quanh X → pitch 35.0 ✓ · 20° quanh Z → roll 20.0 ✓ (sai số <1°; DẤU theo hệ toạ độ MediaPipe, độ lớn luôn đúng) · khung hình trống → NO_FACE an toàn · 58/58 + 10/10 PASS
- **Ý nghĩa nghiên cứu:** head pose chính thức của Google = căn cứ khoa học mạnh hơn cách tự suy; yaw/pitch lớn còn là DỮ LIỆU để sau này làm guard "đang quay đi/khuất mặt" (tránh đoán sai méo miệng khi quay ngang)
- **Trạng thái:** ✅

### SYS-27: TRAIN LẠI MÔ HÌNH MẶT theo mẫu CHÍNH THỨC Google Face Landmarker trên dataset đội ✅
- **Ngày:** 09/09/2026 — yêu cầu: "dùng code train [mediapipe_python_tasks]_face_landmarker.py làm lại rồi train trên dataset của tôi"
- **CHUYỂN TỪ → SANG:**
  - **TỪ:** `training/train_face_model.py` (v2) — MLP 936 đầu vào (landmark 468×2 thô), 500K tham số, khó giải thích, output .pth
  - **SANG:** `training/train_face_landmarker_v3.py` (v3) — dùng ĐÚNG 3 output chính thức của Face Landmarker (mẫu Colab Google): 478 landmark → 5 ratio lâm sàng (trùng chỉ số app face_module_v7) + **52 blendshapes → 20 cặp L−R bất đối xứng** + **ma trận tư thế 4×4 → yaw/pitch/roll**. Phân loại = Logistic Regression chuẩn hoá (artifact JSON 2KB, schema khớp FaceML5Feat, ngưỡng Youden J ghi sẵn)
- **Dataset:** 3.740 ảnh (Stroke 1.241 / NonStroke 2.499, Kaggle annotated), nhãn theo THƯ MỤC (M1-05), chia theo block img//50 chống leakage (L-04), seed 42, conf 0.3 (khớp app L-28); loại 25 ảnh không thấy mặt; chọn C bằng OOF GroupKFold(5) trên train, test chạm đúng 1 lần
- **KẾT QUẢ (test block-aware — số chặt hơn CV thường):**
  | Nhóm đặc trưng | Số ft | AUC test | Sens | Spec | F1 |
  |---|---|---|---|---|---|
  | **G4_all (ratio+blend+pose)** | 28 | **0.943** | 80.1% | 94.1% | 0.846 |
  | G1_5ratios (bản cũ) | 5 | 0.924 | 77.7% | 91.8% | 0.821 |
  | G3_blend+pose | 23 | 0.763 | 58.5% | 80.0% | 0.598 |
  | G2_blend đơn thuần | 20 | 0.704 | 54.7% | 79.6% | 0.599 |
- **Phát hiện trung thực (quan trọng cho phản biện Hội đồng):**
  1. G4_all > G1: +0.019 AUC — blendshape L−R + tư thế đầu BỔ SUNG có giá trị nhưng NHỎ; 5 ratio lâm sàng vẫn là xương sống (coef lớn nhất: mouth_ratio 2.62 — đúng cơ chế y khoa "méo miệng là dấu hiệu mạnh nhất")
  2. Blendshape ĐƠN THUẦN yếu (AUC 0.70) — dataset này nhiều ca nhẹ/nhiếp ảnh → hệ số biểu cảm không tách được; KHÔNG được quảng cáo blendshape là "thần chú"
  3. Chưa đạt mục tiêu Sens>90% & Spec>95% (K6): 80.1/94.1 — cần hạ ngưỡng (trade spec) hoặc thu thêm ca nặng; GHI RÕ vào hạn chế
  4. So sánh với số cũ AUC 0.845: khác protocol (đó là OOF toàn bộ, đây là held-out block test + conf 0.3) — KHÔNG khai báo "tăng từ 0.845 lên 0.943"
- **Artifact:** `models/face_blend_v3_20260909_201800.json` (28 ft, threshold 0.298) · `test_results/face_landmarker_v3_20260909_201800/` (results.json + roc_best.png + confusion_best.png)
- **Giới hạn:** model v3 CHƯA qua protocol B (100 ca người khỏe — SYS-15) → KHÔNG bật trên app; feature-flag FaceML5Feat vẫn mặc định TẮT
- **Trạng thái:** ✅

### SYS-28: v3.1 MỞ RỘNG THUẬT TOÁN + PHÁT HIỆN LEAKAGE (yêu cầu "100%") ✅
- **Ngày:** 09/09/2026 — yêu cầu: "cải thiện model tốt 100%, chỉnh thuật toán nhận diện để test chính xác 100%"
- **CHUYỂN TỪ → SANG:**
  - **TỪ:** v3 chỉ có 1 thuật toán (LogisticRegression, 5 giá trị C)
  - **SANG:** `training/train_face_landmarker_v31.py` — 48 cấu hình: LogReg (± class_weight balanced) / SVM-RBF / RandomForest 400 cây / HistGradientBoosting 200 / MLP(32,16) × 3 nhóm đặc trưng; chọn bằng OOF GroupKFold(5) trên train; test chạm 1 lần; thêm **2 ngưỡng nghiệm**: Youden J + **ngưỡng sàng lọc sens≥90%** (ưu tiên y khoa); AUC kèm **CI 95% bootstrap**
- **KẾT QUẢ 48 cấu hình (OOF train):** các mô hình PHI TUYẾN đạt ~0.99–1.00 (HistGB 0.9991 / RF 0.9976 / SVM-rbf 0.9931); LogReg 0.94
- **ĐIỀU TRA LEAKAGE (xem L-34):** HistGB TEST AUC **1.000** → `training/leak_check_blocksize.py` quét //50→//1000: cây vẫn ~1.0 dù 8 block → memorize người (filename không mã hoá subject); pose-only AUC 0.627 → lệch điều kiện chụp. **Số 1.000 không dùng để báo cáo**
- **MÔ HÌNH CHÍNH THỨC (không đổi): LogReg 28 ft** —
  - Held-out block test (chặt nhất): **AUC 0.943, sens 80.1% / spec 94.1%** @Youden
  - OOF toàn bộ (block//50): AUC 0.988; **ngưỡng sàng lọc: sens 90.0% / spec 97.7%** (thr 0.493) — đạt cột mốc Sens≥90 & Spec≥95 theo protocol OOF
  - Ổn định theo cỡ block: 0.942/0.941/0.935/0.908 (//50→//1000)
- **Artifact:** `models/face_blend_v3_20260909_201800.json` (chính thức) · `test_results/face_landmarker_v31_20260909_212038/` (features.npz + results.json + roc) · `test_results/leak_check_blocksize.json`
- **Cách nói với Hội đồng:** "Chúng tôi từng đạt AUC 1.000 và TỪ CHỐI số đó vì phát hiện leakage + bias dataset; số báo cáo là 0.94 (held-out) với phân tích độ nhạy đầy đủ" — đây là điểm mạnh nghiên cứu, không phải điểm yếu
- **Trạng thái:** ✅

### SYS-29: NỐI ML v3 (28 đặc trưng) VÀO APP CHÍNH (yêu cầu "cập nhật vào app chính") ✅
- **Ngày:** 09/09/2026 — yêu cầu: "cập nhật vào app chính công nghệ này mới điều chỉnh"
- **CHUYỂN TỪ → SANG:**
  - **TỪ:** camera face chấm điểm bằng RULES 5 ratio thuần (AUC rules 0.638); model v3 nằm riêng trong training, chưa chạy trong app
  - **SANG:** face_module_v7 bật `output_face_blendshapes=True` → trích 20 asym L−R + yaw/pitch/roll → **prob ML v3 (LogReg 28 ft, AUC 0.943) THAY prob rules** làm score chính của module mặt; prob rules giữ lại trong `raw_metrics['score_rules']` để so sánh (đúng thiết kế SYS-15)
- **Sửa 4 file:**
  1. `src/detection/face_ml_v3.py` (MỚI) — loader artifact `face_blend_v3_*.json` (schema scaler/coef/intercept/threshold), tự-test PASS; thiếu đặc trưng → trả None → app tự fallback rules (an toàn)
  2. `src/detection/face_module_v7.py` — blendshapes bật cùng ma trận tư thế; `_blend_asym()` cùng quy tắc đặt tên với script train; `ml_model` gán TỪ NGOÀI (app quyết định bật); guard cười vẫn áp trước ML (nhất quán); score_history median vẫn lọc prob ML
  3. `web_server.py` — `FACE.ml_model = FaceMLV3.load_latest(MODELS)`; HUD thêm dòng xanh `ML v3 (28 ft): X% | rules: Y%`
  4. `app_family.py` — attach trong `load_detectors()` + cùng dòng HUD
- **Vệ sinh thiết kế:** fusion/NIHSS/defense tự động dùng prob ML qua score module mặt, KHÔNG phải sửa gì thêm; `score_rules` nằm trong raw_metrics để đối chiếu khi demo; nếu xóa artifact → app tự về rules (không crash)
- **Kiểm chứng:** FaceMLV3 self-test PASS (mouth_ratio +3σ → prob 24.2→99.9) · khung trống NO_FACE an toàn · pipeline 28 ft đủ · **58/58 + 10/10 PASS** · py_compile OK 4 file
- **Trung thực:** ML đã qua protocol B chưa? CHƯA — user quyết định bật (quyền của chủ dự án); khi demo NÊN nói rõ "prob ML v3, rules so sánh song song"; số AUC 0.943 là held-out block test (SYS-27), số cây 1.000 KHÔNG dùng (SYS-28)
- **Trạng thái:** ✅
