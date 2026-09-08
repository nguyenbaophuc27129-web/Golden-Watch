# HƯỚNG DẪN LOGIC TỪNG FILE CODE — HỌC & GHI NHỚ (Golden-Watch PSCS)

> Mục đích: giải thích **logic bên trong từng file** để học thuộc — mỗi file theo khung:
> **Mục đích → Input/Output → Luồng logic → Hàm quan trọng → ⚠️ Bẫy dễ sai**.
> Số liệu niêm yết dưới đây là hằng số thật trong code (ngày 08/09/2026).

## SƠ ĐỒ DỮ LIỆU TỔNG (nhớ 1 dòng)

```
5 MODULE (face/arm/gait/radar/speech) → DefenseEngine.filter (L2/L4)
→ FusionEngine.fuse (trọng số + R1/R2) → NIHSS estimate + CI
→ DefenseEngine.verdict (L3 temporal) → AlertSystem (80/60s)
→ TriageEngine (bệnh viện) → HandoffSystem (QR + PDF)
```

---

## 1. app_family.py — ỨNG DỤNG GIA ĐÌNH (Streamlit, v8.1)

- **Mục đích:** 1 app duy nhất cho người nhà, chia **2 phần: 📷 Camera / 📡 Radar** + phần kết quả hợp nhất.
- **Input:** webcam C270, radar LD2450 (COM hoặc SIM), micro. **Output:** cảnh báo + QR/PDF bệnh viện.
- **Luồng logic:**
  1. `init_session()` — khởi tạo TOÀN BỘ `st.session_state` (patient, running, thresholds, `cam_error=None`, `radar_history=[]`, `face_ml` (FaceML5Feat.load_latest), `face_ml_enabled=False`).
  2. `main()` — 3 nhóm: `## 📷 PHẦN 1 — CAMERA` (tabs: Giám sát camera, Kiểm tra nói) → `## 📡 PHẦN 2 — RADAR LD2450` (tab Giám sát radar) → `## 🧩 KẾT QUẢ HỢP NHẤP` (tabs: Báo động, Kết quả & NIHSS, Bệnh viện & Báo cáo).
  3. `monitor_fragment()` — vòng lặp 3s: face → arm → gait (camera) + radar.read_targets() → lưu `radar_history` (cột: Thời điểm/Trạng thái/Dịch chuyển/Bất động/Chế độ, giữ 120 dòng) → Defense → Fusion → Alert. Face ML: nếu `face_ml_enabled` → `predict(raw_metrics)` **thay** score/status của rules (giữ `rules_prob` để đối chiếu).
  4. Sidebar: form bệnh nhân (key `'patient_form'`), ngưỡng, checkbox face ML, test còi.
- **Hàm quan trọng:** `init_session`, `main`, `monitor_fragment`, `tab_monitor`, `tab_speech`, `tab_radar`, `tab_alert`, `tab_results`, `tab_hospital`.
- **⚠️ Bẫy dễ sai:**
  - Streamlit **form key trùng session_state key** → lỗi `StreamlitValueAssignmentNotAllowedError` (đã sửa: `'patient'`→`'patient_form'`).
  - Đọc `ss.<attr>` chưa khởi tạo trong `init_session` → crash lần render đầu (đã thêm `ss.cam_error=None`).
  - `list_ports()` radar trả **tuple (dev, desc)**, không lưu `self.port` → phải giữ tên cổng từ label.

---

## 2. src/detection/face_module_v7.py — MÉO MẶT (MediaPipe rules)

- **Mục đích:** đo méo mặt theo 5 tỉ lệ từ 478 landmark → prob 0–100 (luật, không ML).
- **Input:** frame BGR. **Output:** `{score, status, metrics{5 đặc trưng thô}, nihss_item4, ...}`.
- **Luồng logic:** `process_frame` → MediaPipe FaceLandmarker (`models/face_landmarker_v2.task`) → `_extract_landmarks` (478×3) → 5 hàm tính:
  `_calculate_mouth_asymmetry` (đối xứng khóe miệng), `_calculate_eye_deviation` (sụp mí/viền mắt), `_calculate_face_tilt` (nghiêng trục), `_calculate_nasolabial_asymmetry` (rãnh mũi má), `_calculate_forehead_asymmetry` (nếp trán) → `_calculate_overall_score` (trộn có trọng số → 0–100) → `_map_to_nihss_item_4`.
- **Ngưỡng app:** score ≥30 = WARNING, ≥60 = DANGER.
- **Kết quả thật:** AUC OOF 0.638 (rules) — yếu vì luật tuyến tính trên tọa độ thô.
- **⚠️ Bẫy:** `NO_FACE` khi không thấy mặt (fusion coi là MISSING, không phải 0); file `.task` phải là bản thật 3.670KB — stub 8KB gây `NO_DETECTOR` mãi mãi.

## 3. src/detection/face_ml_5feat.py — MÉO MẶT ML (5 đặc trưng)

- **Mục đích:** logistic regression trên 5 đặc trưng → AUC OOF **0.845** (thay rules khi bật cờ).
- **Input:** `raw_metrics` dict từ face_module_v7. **Output:** prob 0–100 hoặc `None` (thiếu đặc trưng).
- **Luồng logic:** `load_latest(models_dir)` glob `models/face_asym_v2_5feat_*.json` lấy mới nhất → artifact chứa `coef, intercept, mean, std, threshold, features` → `predict`: chuẩn hoá z-score → sigmoid → ×100.
- **⚠️ Bẫy:** bật ML trong app nhưng artifact không tồn tại → phải tự tắt cờ (app check `face_ml is not None`); ML học trên Kaggle block-split — chưa phải dữ liệu người VN thật.

## 4. src/detection/arm_module.py — YẾU TAY (YOLOv8n-pose)

- **Mục đích:** phát hiện 1 tay yếu/rơi xuống khi giơ 2 tay (NIHSS item 5).
- **Input:** frame (người giơ tay). **Output:** `{arm_prob, weak_arm, nihss_item5, metrics}`.
- **Luồng logic:** `detect_arm_weakness(frame)`:
  1. YOLOv8n-pose → nếu không người → `NO_PERSON`.
  2. `keypoints.xy[0]` (17,2) **+ `keypoints.conf[0]` → column_stack thành (17,3)** ← bản sửa M3-06.
  3. `extract_arm_keypoints`: vai(5,6)-khuỷu(7,8)-cổ tay(9,10).
  4. `extract_arm_features`: `calculate_arm_angle` (góc vai-khuỷu-cổ tay), độ cao cổ tay 2 bên, chênh lệch trái–phải.
  5. `_rule_based_score` (chênh góc/độ cao → prob) → `_identify_weak_arm` → `_map_to_nihss`.
- **⚠️ Bẫy (đã dính):** `keypoints.xy` chỉ (17,2) — code cũ đọc `[2]` → **IndexError** khi có người thật trong khung; webcam gần mặt trước đây nên không crash = bug ẩn. Fix đã khóa bằng test SYS-14b số 3.

## 5. src/detection/gait_module.py — DÁNG ĐI (2 lớp riêng biệt)

- **Lớp A `GaitAbnormalityDetector` (dữ liệu PhysioNet — dùng cho eval):**
  - Input: chuỗi thời gian 8 đặc trưng lực bàn chân (force plate). Luồng: `load_gait_data` → `extract_gait_features` → ML (artifact) + `_rule_based_score` → `_map_to_nihss`. Kết quả LOSO: Sens 94.1, Spec 88.3, AUC 0.879.
- **Lớp B `GaitPoseDetector` (camera app):** `detect_gait_from_frame` (YOLO pose) → bộ đệm khung → `_calculate_gait_metrics` (6 số đo tư thế) → `_calculate_gait_score`.
- **Quyết định kiến trúc M4-07:** model PhysioNet (8 chiều lực chân) **KHÓA cứng** khỏi camera (6 chiều pose) — không cùng không gian đặc trưng, ghé vào là pseudo-science. Camera chỉ dùng luật, eval chưa có.
- **⚠️ Bẫy:** nhầm 2 lớp là 1; silhouette gait 0.681 (2 cụm tách được) — dùng con số này giải thích tại sao gait khả thi.

## 6. src/detection/radar_module.py — RADAR LD2450 (ngã/bất động)

- **Mục đích:** đọc frame nhị phân LD2450 → mục tiêu (x,y,tốc độ) → phát hiện ngã/bất động lâu.
- **Input:** bytes từ serial 256000 baud (hoặc SIM). **Output:** `{fall_prob, movement_cm, still_seconds, targets}`.
- **Luồng logic:**
  - `LD2450Parser.feed(data)`: đệm bytes → tìm header `AA FF 03 00` → `parse_frame` tách 4 mục tiêu.
  - `RadarModule.__init__(port, baud=256000, simulation=None)` → `_connect` (thất bại → tự rơi SIM nếu cho phép).
  - `read_targets()` 1 nhịp; `analyze(duration_s=10, sample_hz=2)`: gom chuỗi → **bất động** = dịch chuyển < ngưỡng suốt `still_seconds`; **ngã** = dịch chuyển đột biến + sau đó bất động.
  - `set_audio_flag(abnormal)`: nhận cờ tiếng nói bất thường từ app (đa mô hình).
  - `list_ports()` → [(dev, desc)] — chỉ trả danh sách.
- **⚠️ Bẫy:** cổng COM trên máy hiện chỉ là Bluetooth → **không có radar thật** phải chạy SIM (app có checkbox); người dùng tự test phần cứng sau (còn [ĐIỀN] trong báo cáo).

## 7. src/detection/speech_module_v2.py — NÓI KHÓ (TORGO, 48 đặc trưng)

- **Mục đích:** phát hiện nói đớ/khó (dysarthria — NIHSS item 10).
- **Input:** audio 16kHz. **Output:** `{speech_prob, wpm, transcript, nihss_item10}`.
- **Luồng logic (`predict_dysarthria`):**
  1. `record_audio(5s)` → `detect_voice_activity` (VAD theo năng lượng, frame 30ms) → `select_best_window` (cửa sổ 5s tốt nhất, hop 0.5s, median-3 chống nhiễu).
  2. `extract_features(audio)` → **trả TUPLE `(features_array_48, features_dict)`** — dict có `pitch_mean`, jitter, shimmer, HNR... (librosa).
  3. `transcribe_speech` (Vosk) → `calculate_wpm` → `_rule_based_score` (WPM thấp + pitch/jitter lệch baseline) và ML `DysarthriaClassifier` (MLP 48→256→128→64→2, dropout 0.5) → trộn → `_map_to_nihss`.
  4. Baseline cá nhân: `set_baseline/save_baseline/load_baseline/calibrate_baseline(30s)`.
- **Kết quả thật (TORGO 55 session):** Sens 96.3, Spec 85.7, AUC 0.992 — nhưng **chưa tiếng Việt** (đang thu L-02).
- **⚠️ Bẫy (đã dính 2 lần):** gọi `extract_features` mà quên unpack tuple; numpy 2.5 làm hỏng numba→librosa mất `pitch_mean` → **ghim numpy 2.3.4**.

---

## 8. src/fusion/fusion_engine.py — TRỌNG SỐ + LUẬT R1/R2

- **Mục đích:** gộp 5 module → 1 điểm 0–100 + mức rủi ro.
- **Hằng số PHẢI NHỚ:**
  - Trọng số: **face .20 · speech .20 · arm .30 · gait .15 · radar .15** (arm cao nhất vì dấu hiệu khu tr mạnh nhất).
  - **R1:** ≥2/3 tín hiệu FAST (face, speech, arm) có prob ≥50 → **EMERGENCY** bất kể điểm gộp.
  - **R2:** bất kỳ module ≥80 → **WARNING** tối thiểu.
  - Băng: score ≥70 EMERGENCY · 50–69 WARNING · <50 SAFE.
- **Luồng `fuse(module_results)`:** `_normalize_module` từng module (status rỗng ∈ {NO_FACE, NO_PERSON, NO_POSE, NO_DATA, NO_SPEECH, NO_DETECTOR, NO_MODEL, ERROR} → MISSING, không cộng điểm) → tính điểm có trọng số trên các module CÓ dữ liệu (chuẩn hoá lại tổng trọng số) → áp R1/R2 → NIHSS → `_classify_severity` → `_recommendation` → đẩy `(score, time)` vào `history` (100 mục) cho `get_trend(window=5)`.
- **⚠️ Bẫy:** module MISSING làm TỔNG TRỌNG SỐ thay đổi — điểm gộp không so sánh trực tiếp khi số module khác nhau; R1 có thể chuyển SAFE→EMERGENCY.

## 9. src/fusion/nihss_estimator.py — NIHSS 4 ITEM (tối đa 13)

- **Mục đích:** ước lượng điểm NIHSS từ prob module (item 4 mặt/5 tay/6 chân/10 nói).
- **Bảng `PROB_TO_ITEM` (prob ≥ ngưỡng → điểm):**
  - item4 mặt: (80→3) (55→2) (30→1) (0→0) — max 3
  - item5 tay: (85→4) (70→3) (50→2) (25→1) (0→0) — max 4
  - item6 chân: giống item5 — max 4
  - item10 nói: (60→2) (30→1) (0→0) — max 2
- **`prob_to_score` quét TỪ CAO XUỐNG THẤP** — trả điểm band đầu tiên thoả.
- **`estimate_nihss`:** module status rỗng → `items_missing` (không bịa 0). Trả `{items, total, max_possible:13, items_missing, note}`.
- **`calculate_nihss_ci(n_iter=1000, noise=10, seed=42)`:** Monte Carlo — mỗi lần cộng nhiễu Gauss σ=10 vào prob, clip [0,100], chấm lại → margin = 1.96×std → trình bày "8 ± 2".
- **⚠️ Bẫy:** đây là UỚC LƯỢNG HỖ TRỢ — note trong output ghi rõ "bác sĩ chấm NIHSS chuẩn"; chưa có ground truth 50 video (SYS-18) nên KHÔNG được nêu MAE/R².

## 10. src/fusion/triage_engine.py — CHỌN BỆNH VIỆN

- **Mục đích:** NIHSS + triệu chứng → mức nặng + bệnh viện có stroke unit gần nhất.
- **Luồng:** `estimate_subtype(symptoms, module_results)` (đại thể LACS/PACS/...) → `classify_severity(nihss_total, trend)` → `load_hospitals` đọc CSV → `recommend_hospital(severity, subtype)` → `triage(fusion_result, nihss_total, ...)` gói tất cả.
- **⚠️ Bẫy:** CSV bệnh viện là dữ liệu mẫu — phải ghi rõ trong báo cáo là demo, không phải định vị thật.

## 11. src/fusion/validation_metrics.py — THƯ VIỆN ĐO (không phụ thuộc sklearn)

- **Mục đích:** tự viết metrics để test không lệch thư viện: `pearson_r, mae, confusion_matrix, sensitivity, specificity, f1_score, roc_points, find_youden, paired_sign_test, evaluate_nihss_study, evaluate_detection_study`.
- **Nhớ:** `find_youden` = ngưỡng maximize (Sens + Spec − 1); `paired_sign_test` so 2 mô hình trên CÙNG mẫu (không cần phân phối).
- **⚠️ Bẫy:** hàm ở đây tính thô (dùng trong 58 test) — số trong báo cáo lấy từ `training/evaluate_all_metrics.py` (có Wilson CI).

---

## 12. src/defense/defense_engine.py — 4 TẦNG CHỐNG BÁO GIẢ (H2)

- **Mục đích:** giảm false alarm mà không che đột quỵ thật.
- **4 tầng (nhớ L1→L4):**
  - **L1 Calibration:** `calibrate(wpm, face_asym, arm_asym, ...)` — baseline người thật khi khoẻ.
  - **L2 Context:** `update_context(motion_intensity, talking)` → state MOVING/TALKING/REST; `CONTEXT_SUPPRESS[context]` = tập module bị "đánh dấu" (giữ prob, gắn `defense_note`, KHÔNG xoá — xoá có thể che đột quỵ thật khi đang vận động).
  - **L3 Temporal** (`verdict(fused_score)`): score <50 → không alert; lấy mẫu trong **30s** gần nhất (`PERSISTENT_SECONDS`), cần **≥60% mẫu ≥50** mới PERSISTENT (alert); <3 mẫu → "warming up" CHO QUA (an toàn phía quá cảnh báo khi khởi động); transient → chặn, `stats['suppressed_L3']+=1`. Trend 300s: Δ>10 → WORSENING (`WORSEN_DELTA`).
  - **L4 Adaptive:** `_adaptive_floor` — nếu đã calibrate và prob nằm **vùng xám [30,40)** → trừ 5 điểm (baseline cá nhân đáng tin hơn).
- **`filter(module_results)`** → `(clean, audit)`; audit đếm `suppressed_L2 / adapted_L4`; `get_stats()` trả tổng.
- **⚠️ Bẫy:** thứ tự đúng là L2/L4 trong `filter` (trước fusion), L3 trong `verdict` (sau fusion) — đảo thứ tự là sai thiết kế.

## 13. src/alerts/alert_system.py — CÒI + LOG

- **Hằng số:** ngưỡng **80**, cooldown **60s**.
- **Luồng:** `process_fusion_result` (score ≥80 → `send_alert`, tự động) / `send_alert(score, message, level, source)` → kiểm cooldown → `_buzzer(level)` (winsound, EMERGENCY khác WARNING) → `_log_event` (JSONL vào log_dir) → `_app` (Zalo nếu đã `configure_zalo`) → `get_recent_alerts`, `get_stats`.
- **⚠️ Bẫy:** cooldown chặn lặp cùng mức — alert WARNING 55 trong AppTest là bắn thủ công (source='manual') nên không qua ngưỡng 80; dùng `send_alert` khi test, đừng kỳ vọng auto ở 55.

## 14. src/handoff/handoff_system.py — QR + PDF BỆNH VIỆN

- **Luồng:** `add_event(tag, description, data)` (nhật ký sự kiện trong phiên) → `build_report(fusion_result, nihss_result, triage_result, ...)` (dict chuẩn) → `save_report` (JSON) → `generate_qr` (PNG — nội dung tóm tắt + path file JSON) → `generate_pdf` (reportlayout: bệnh nhân, thời gian, điểm, NIHSS, bệnh viện) → `qr_data_url` (nhúng vào Streamlit).
- **⚠️ Bẫy:** QR chứa đường dẫn MÁY NÀY — file JSON phải đi kèm khi chuyển máy; không có mạng vẫn chạy (đúng yêu cầu local-first).

---

## 15. src/test_all_metrics.py — 58 UNIT TEST (bộ test chính, KHÔNG phải pytest)

- **Mục đích:** khóa hành vi toàn hệ thống. Chạy: `python src/test_all_metrics.py` → "TỔNG KẾT: 58/58 PASS".
- **Cấu trúc:** `check(section, name, fn)` bọc từng test + `assert_eq/assert_true/assert_close` tự viết (không phụ thuộc pytest).
- **10 nhóm (chữ cái A–K):** A speech (11: A1–A11) · B fusion (8) · C alert (4) · D models (4) · E artifacts (2) · F defense (6) · G radar (6) · H nihss (4) · I triage (4) · J handoff (3) · K validation (6) = 58.
- **⚠️ Bẫy:** test dùng audio tổng hợp `make_tone/make_silence`; sau MỌI thay đổi pip (torch/numpy) phải chạy lại cả 58 + 10 pre-flight.

## 16. src/test_sys14b_regression.py — 10 PRE-FLIGHT TRƯỚC DEMO

- **Mục đích:** chặn demo hỏng — kiểm tra môi trường "thật" (file >1MB, chạy trên ảnh thật, không stub).
- **10 check:** `.task` >1MB · face chạy ảnh thật ≠ NO_DETECTOR · arm khung trống + khung người không crash · speech unpack `feat, fd` đúng và có `pitch_mean` · gait load · radar load.
- **Thoát mã 1 = KHÔNG demo.** Chạy trước mỗi buổi nghiệm thu (KICH_BAN_NGHIEM_THU.md bước S0).

---

## 17. training/ — HUẤN LUYỆN & ĐÁNH GIÁ

### train_face_model.py + evaluate_face_rules.py + test_face_signal.py
- Chuỗi: chấm điểm rules trên Kaggle (478 ảnh, block-split) → logistic + MLP 5 đặc trưng GroupKFold(5) → xuất `models/face_asym_v2_5feat_*.json` (AUC 0.845) + CSV eval để chart đọc lại.

### train_gait_model_v2.py — LOSO 15 subject
- **Hàm quan trọng:** `load_subjects(path, detector)` (nhóm window theo subject), `train_fold(Xtr, ytr, Xte, device)` (chuẩn hoá fit trên train, MLP nhỏ seed 42), `DEFAULT_GAIT_PATH`, `SEED=42`.
- Ngưỡng Youden chọn TRÊN OOF (như plot script) — không được chọn trên tập test.
- **⚠️ Bẫy:** quên seed → AUC 0.879↔0.890 (biến thiên tự nhiên, CI bootstrap che được — ghi chú trung thực).

### train_speech_torgo.py
- TORGO 55 session → 48 đặc trưng → MLP → `module2_test_*.json` (Sens 96.3 / AUC 0.992, session-level split).

### evaluate_all_metrics.py — BỘ ĐO CHÍNH CHO BÁO CÁO
- 4 evaluator (face_rules, face_ml, gait LOSO, speech) + `wilson(k,n)` + bootstrap AUC + silhouette + `missing_data_audit` (ghi THIẾU: arm, NIHSS-regression, radar, speech-VN). Output `test_results/metrics_pack_*`.

### plot_accuracy_charts.py — 5 BIỂU ĐỒ A–E
- `load_data()` dựng lại OOF prob (seed 42 đầu hàm) → A ROC gộp · B Precision–Recall (kèm tỷ lệ dương nền) · C bar 5 metrics ± **Wilson bất đối xứng** · D sens/spec theo ngưỡng + Youden · E forest AUC ± bootstrap CI. Output `charts_accuracy_*` + `chi_muc.json`.
- **⚠️ Bẫy đã sửa:** wilson phải trả `(điểm, lo, hi)` — bar dùng biên dưới làm điểm là SAI; thiếu seed toàn cục trước gait LOSO làm số lệch.

---

## 18. module5/ — PROTOTYPE

### prototype_benchmark.py
- Đo latency thật RTX 3050: face 15.3ms · arm 27.7ms · gait 17.4ms (CPU)/20.5 (CUDA) · fusion ~0 → tổng ≈**81ms** so chu kỳ app 3000ms. Warmup 3 chu kỳ, xuất JSON + PNG.

### test_radar_hardware.py
- Script test LD2450 qua COM (người dùng tự chạy khi cắm radar) — in mục tiêu thô để kiểm dây/baud.

---

## 19. BẢNG GHI NHỚ 30 GIÂY (trước khi bảo vệ)

| Câu hỏi | Trả lời |
|---|---|
| Trọng số fusion? | face .20, speech .20, **arm .30**, gait .15, radar .15 |
| R1/R2? | R1: ≥2/3 FAST prob≥50 → EMERGENCY; R2: 1 module ≥80 → WARNING |
| 4 tầng chống báo giả? | L1 Calibration, L2 Context, L3 Temporal (30s/60%), L4 Adaptive (vùng xám 30–40 trừ 5) |
| NIHSS? | 4 item (mặt 3, tay 4, chân 4, nói 2) tối đa **13**, CI Monte Carlo 1000 lần "8 ± 2" |
| Alert? | ngưỡng **80**, cooldown **60s**, còi khác nhau theo mức |
| Vì sao face ML 0.845 mà rules 0.638? | tọa độ thô không tách lớp (AUC .555, silhouette .049) nhưng 5 tỉ lệ đặc trưng tách được (d=0.759) — bài học biểu diễn đặc trưng |
| Tại sao gait không nối camera? | M4-07: 8 đặc trưng lực chân ≠ 6 pose — không cùng không gian, nối là giả khoa học |
| Số mạnh nhất? | Speech AUC 0.992 (nhưng chưa tiếng Việt — trung thực) |
| Điểm yếu khai báo? | arm chưa eval dữ liệu thật · NIHSS chưa ground truth · radar chờ test cứng |
