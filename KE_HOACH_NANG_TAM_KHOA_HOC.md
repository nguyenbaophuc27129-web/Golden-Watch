# KE HOACH NANG TAM KHOA HOC — TỪ "TÍCH HỢP" SANG "CHUẨN PHƯƠNG PHÁP QUỐC TẾ"

> v2.0 — 10/09/2026 (CHỐT: KHÔNG thu dữ liệu người ngoài, KHÔNG triển khai
> 3 hộ gia đình — chỉ dùng dataset công khai ĐÃ CÓ + dữ liệu tự thân đội +
> tối đa 1 hộ nhà thành viên). v1.0 — 10/09/2026. Mục tiêu: trả lời được
> chất vấn khó nhất của hội đồng chuyên ngành y/KTYS: *"Đây chỉ là tích hợp,
> chưa có phát hiện y khoa cấp thế giới."*
>
> Trả lời đúng KHÔNG PHẢI là "chúng tôi có phát hiện y khoa mới" (không thể —
> không có bệnh nhân thật), mà là: **hầu hết AI y tế công bố quốc tế chỉ dừng
> ở internal validation — điều TRIPOD+AI (BMJ 2024;385:e078378) cảnh báo.
> Dự án này đi ĐỦ 3 BẬC THANG KIỂM ĐỊNH quốc tế + 2 đóng góp phương pháp.**

## v3.0 — 12/09/2026: GÓI NÂNG CẤP "HỆ GIÁM SÁT TỰ TIN" (A+B+D) — CHỐT CHO GIẢI QG

> Quyết định 12/09 sau khi đối chiếu hướng nâng cấp. Câu chuyện duy nhất cho
> báo cáo + phỏng vấn: **"Golden-Watch 3.0 — hệ giám sát TỰ TIN: (A) biết mình
> không biết — abstention; (B) coi DÒNG THỜI GIAN chứ không phải khoảnh khắc —
> early-warning; (D) HAI MẮT CẢM BIẾN đối chéo radar↔camera."**
> NGUYÊN TẮC: chỉ là LỚP BỔ SUNG qua flag có thể tắt (fallback = hệ thống cũ
> đã eval); model ĐÓNG BĂNG; không sửa ngưỡng đã chấm; seed 42.

### A — UNCERTAINTY-AWARE FUSION + ABSTENTION ⭐ (dùng OOF có sẵn, 0 data mới)
- **Vấn đề:** AI y tế hiện nay thường "bắt buộc đoán quyết" — prob 47% vẫn phải
  chọn NORMAL/WARNING. TRIPOD+AI khuyến nghị báo độ bất định. Giải pháp:
  module BẤT ĐỊNH → trạng thái mới `NEEDS_CHECK` (người xác nhận) thay vì báo giả.
- **Làm:** (1) Platt-calibrate prob từng module trên OOF ĐÃ LƯU (speech
  `speech_speaker_loso_*/oof_predictions.csv`; face tính lại OOF per-image từ
  cache `_face_v3_features_cache.npz`; gait LOSO PhysioNet chạy lại — 15 subject,
  rẻ); (2) uncertainty = phương sai giữa cửa sổ (speech đã có spread 3 cửa sổ,
  face/gait tương tự); (3) luật abstain: |prob−ngưỡng| < τ HOẶC spread cao.
- **Đo:** risk–coverage curve, selective-AUC, FPR@TPR cố định có/không abstain
  — per-module + system-level qua replay Protocol B. Figure: `abstention_curve.png`.
- **File mới:** `src/fusion/uncertainty.py` (thuần hàm) + `training/eval_abstention.py`.
  FusionEngine thêm flag `abstain=True` (mặc định tắt ở app demo nếu muốn giữ cũ).

### B — TEMPORAL EARLY-WARNING (nâng cấp L3 từ "luật 30s" thành thống kê)
- **Cơ sở:** DefenseEngine L3 đang là luật cửa sổ 30s/1p/5p + `_trend()` thô.
  Nâng thành **EWMA + CUSUM change-point** trên chuỗi fused score + **vận tốc
  suy giảm** (slope/30s) → trạng thái riêng `DETERIORATING` (suy giảm tiến triển)
  khác với EMERGENCY đơn điểm — đúng bản chất "đồng hồ vàng là giờ chứ không
  phải giây".
- **Đo:** (a) replay harness Protocol B với FakeClock — độ trễ phát hiện + FPR;
  (b) giả lập suy giảm tiêm vào log 72h lab (ghi rõ "simulated"); (c) FAR/24h
  + uptime từ Protocol F. Figure: `early_warning_trajectory.png`.
- **File mới:** `src/defense/early_warning.py` (numpy thuần) + `training/eval_early_warning.py`.

### D — CROSS-MODAL SENSOR-CHECK (radar↔camera đồng thuận)
- **Luật:** chuyển động radar (Doppler/nhãn) vs camera (YOLO pose) trong cùng
  cửa sổ 2s PHẢI đồng thuận; lệch → `defense_note` + giảm ĐỘ TIN CẬY hiển thị
  (giữ nguyên điểm — đúng bài học L-08: không sửa điểm để không che đột quỵ thật).
- **Demo mạnh:** che camera → radar vẫn bắt ngã; người rời khung → radar xác nhận
  không ngã → bớt báo giả.
- **Đo:** radar hardware 10 kịch bản M5-02 + Protocol B — % báo giả được đánh dấu
  mà không đụng true positive. File mới: `src/defense/cross_check.py`.

### TIMELINE v3.0 (12/09 → 05/10) — thay timeline v2.0
| Ngày | Việc |
|---|---|
| 12–14/09 | **A**: uncertainty.py + eval_abstention.py → figure abstention |
| 15–17/09 | **B**: early_warning.py + replay eval → figure trajectory |
| 18–20/09 | **D**: cross_check.py + radar thật 10 kịch bản (M5-02) + quay E3 arm |
| 21–23/09 | Protocol B 100 tình huống × ablation (abstention/cross-check là 2 cấu hình mới) |
| 24–28/09 | SYS-18 NIHSS (50 video) + E2 speech VN |
| 25/09–03/10 | Protocol F: 72h lab (FAR/24h + trend stability, single-site) |
| 01–04/10 | Hồ sơ: báo cáo có mục "3 trụ cột tự tin" + 5 figures + video + poster |
| 05/10 | NỘP HỒ SƠ |

### VỊ TRÍ TRONG BÁO CÁO
- Mục Thiết kế & PP: kiến trúc 5 module → 4 lớp defense → fusion → **3 trụ cột
  tự tin** → triage. Mục Tiến hành: bậc thang 3 bậc + BlockSweep + ablation
  (giữ nguyên) + **5 figures mới** (abstention, trajectory, cross-check, trend-72h,
  defense curve).

---

## v2.0 — CHIẾN LƯỢC MỚI: "Khai thác SÂU dataset có sẵn" (không cần ai ngoài đội)

Kiểm kê `fga_project/data/datasets/`: face 7,490 file (ĐANG DÙNG) · gait
PhysioNet (ĐANG DÙNG, LOSO đủ chuẩn) · speech TORGO 1.7GB 17,635 file
(ĐANG DÙNG nhưng CHƯA HẾT chiều) · **pose RỖNG** (arm giữ nhãn synthetic).

### NÂNG CẤP S-1 (QUAN TRỌNG NHẤT): Speech split theo NGƯỜI THẬT + LOSO
- Phát hiện 10/09: `train_speech_torgo.py` split theo `hash(speaker_dir)`
  mà speaker_dir = **session×mic** (`wav_arrayMic_F03S01` ≠
  `wav_headMic_F03S01`) → cùng người F03 nằm ở CẢ train lẫn test (khác mic).
  Số 0.992 là "session-level", chưa phải người-level.
- Việc: viết eval LOSO theo người thật (strip prefix mic + hậu tố Sxx →
  F01/F03/FC01...), model đóng băng hoặc retrain fold-by-fold → AUC
  người-level + CI. Kết quả dù lên hay xuống đều là BẰNG CHỨNG KHOA HỌC:
  xuống = phát hiện leakage thứ 2 (kể chuyện như face HistGB); giữ cao =
  khẳng định độ mạnh thật.
- Kèm: **cross-mic eval** (train headMic → test arrayMic) = kiểm chứng
  "không phụ thuộc micro" — đúng bài toán demo thật.

### NÂNG CẤP S-2 (công nghệ, chọn 1 nếu còn thời gian sau S-1):
- **openSMILE eGeMAPSv02** (88 features chuẩn trong tài liệu rối loạn nói,
  `pip install opensmile`) so với 48 features tự trích — chỉ so sánh,
  production giữ feature hiện có nếu không tốt hơn rõ.
- **wav2vec2-base FROZEN embeddings + LogisticRegression head**: hiện đại,
  chạy offline 1 lần trên 1.7GB (RTX 3050 đủ), chống overfit vì head tuyến
  tính — NHỚ bài học HistGB: head phải đơn giản + split người thật.
- **SpecAugment / speed-perturb** augmentation cho n nhỏ.

### NÂNG CẤP F-1: Face mirror-asymmetry features (v4, tùy chọn)
- Đặc trưng "bất đối xứng gương": lật gương landmark đã align (Procrustes)
  → vector chênh |P − mirror(P)| — chuẩn trong tài liệu facial palsy.
- Thêm vào LogReg (KHÔNG cây) giữ nguyên block-split → so với v3 0.943.
- Kèm **multi-seed (10 seeds)** → AUC 0.943 ± SD (ổn định để báo cáo).

### THAY ĐỔI CÁC PROTOCOL v1:
- **E1 face external** → bỏ thu 20 người. Thay: (a) S-1/S-2/F-1 trên dataset
  có sẵn; (b) nếu còn thời gian: đội TỰ quay 4–6 người × 2 điều kiện
  (biểu mẫu tự nguyện 1 trang cho thành viên/người nhà ruột) — nhãn n nhỏ
  ghi rõ "smoke test".
- **E2 speech VN** → tùy chọn VIVOS công khai (HuggingFace `load_dataset("vivos")`,
  ~15h đọc chuẩn tiếng Việt) đo FPR tiếng Việt; không tải được → giữ
  limitation L-02 như hiện tại.
- **E3 arm** → bỏ dataset ngoài; đội tự quay 3–5 người (tự thân) mô phỏng
  thả tay — eval thật ĐẦU TIÊN của arm, n nhỏ ghi rõ.
- **SYS-18 NIHSS 50 video** → GIỮ NGUYÊN (100% công khai trên YouTube).
- **Protocol F field** → 1 hộ (nhà thành viên) × 5 ngày HOẶC lab 72h chạy
  `web_server.py` liên tục: FA/24h, uptime, crash — ghi nhãn trung thực
  "single-site".
- **Prototype**: BOM hiện tại (laptop RTX3050 + C270 + 1×LD2450 ≈1,01 triệu)
  là ĐỦ và là ĐIỂM CỘNG tính cộng đồng — không cần thêm phần cứng; vỏ gắn =
  hộp nhựa/gỗ đơn giản (~100k đã có trong BOM).



## BẬC THANG KIỂM ĐỊNH (khung trình bày trong báo cáo — mục 4 "Tiến hành")

| Bậc | Tên quốc tế | Trạng thái | Bằng chứng |
|---|---|---|---|
| 1 | **Internal validation** (out-of-fold/LOSO/block-split + CI) | ✅ XONG | face v3 AUC 0.943 held-out block; gait LOSO 0.879; speech TORGO 0.992; Wilson CI mọi tỷ lệ |
| 2 | **External validation** (dữ liệu MỚI, khác nguồn/nhà máy cảm biến/điều kiện) | ⬜ THU 15–30/09 | Protocol E1/E2/E3 dưới đây |
| 3 | **Prospective / field study** (dùng thật tại nhà) | ⬜ THU 25/09–03/10 | Protocol F: 3 hộ × 3 ngày |

Plus 2 đóng góp phương pháp: **BlockSweep** (✅ figure xong) + **Defense ablation** (⬜ gắn vào Protocol B).

---

## ĐÓNG GÓP #1 — GIAO THỨC BLOCKSWEEP (✅ HOÀN THÀNH)

- Script: `training/leak_check_blocksize.py` · Figure: `test_results/blocksweep_leakage_diagnostic.png`
- Nội dung: dò leakage bằng quét kích thước block GroupKFold × đối chiếu họ mô
  hình. **Mô hình tuyến tính ổn định (0.91–0.94 mọi block) = tín hiệu thật;
  mô hình cây ≈ 1.000 kể cả block 1000 (8 block) = memorize người.**
- Cách trình bày: *"Khi mô hình đạt AUC 1.000, thay vì khoe, chúng tôi COI ĐÓ
  LÀ CHỈ SỐ LỖI và dựng giao thức chứng minh nó là leakage."* Khớp TRIPOD+AI
  mục "chống overoptimism". Đặt tên trong báo cáo: **Giao thức BlockSweep**.
- Trả lời phỏng vấn: LogReg tổng quát vì không có đủ độ phức tạp để nhớ
  từng người; cây sâu đủ khả năng ghi nhớ filename→subject dù block 1000.

## ĐÓNG GÓP #2 — DEFENSE ABLATION (gắn vào Protocol B, 21–23/09)

- Trong 100 tình huống người khỏe (Protocol B), chạy LẶP 4 lần với 4 cấu hình:
  `L1-only` → `+L2` → `+L3` → `+L4` (bật/tắt tầng qua cờ DefenseEngine).
- Đo: FPR (EMERGENCY giả) + tổng báo giả ở từng cấu hình → 1 đường cong
  "mỗi lớp giảm bao nhiêu % báo giả".
- DefenseEngine đã có `stats` từng lớp sẵn (`suppressed_L2/L3`, `adapted_L4`).
- Sản phẩm: figure `defense_ablation_curve.png` + dòng trong báo cáo:
  "kiến trúc 4 lớp không phải tuyên bố thiết kế — là kết quả đo được".

---

## PROTOCOL E1 — FACE EXTERNAL VALIDATION (thu mới, KHÁC Kaggle) — 15–17/09

- Đối tượng: ≥20 người (ưu tiên người cao tuổi + người quen khác độ tuổi/giới).
- Điều kiện ×2 mỗi người, webcam C270 thật, ánh sáng phòng thật:
  1. **Bình thường**: ngồi tự nhiên 30s (3 chu kỳ phân tích).
  2. **Mô phỏng méo mặt**: cố tình méo 1 bên miệng + nhắm nhẹ 1 mắt 30s
     (chọn 1 bên ngẫu nhiên mỗi người, ghi lại bên nào).
- Chạy: `face_module_v7` + ML v3 (artifact v3, KHÔNG retrain — external nghĩa
  là model đóng băng). Nhãn ở mức NGƯỜI×ĐIỀU KIỆN.
- Số cần: AUC external + sens/spec @thr 0.298 + so với Kaggle 0.943.
- Kỳ vọng trung thực: external THẤP HƠN internal (domain shift ánh sáng/độ
  phân giải) — báo cả 2 số, chênh lệch chính là nội dung khoa học.
- Lưu: `test_results/external_face_<ts>/` (CSV per-frame + JSON tổng).

## PROTOCOL E2 — SPEECH TIẾNG VIỆT (L-02/M2-11) — 26–28/09

- ≥20 người × đọc 3 câu chuẩn tiếng Việt (in sẵn, cùng câu cho mọi người) +
  1 lượt **nói líu mô phỏng** (đọc câu đó với lưỡi cắn/đọc nhanh méo tiếng).
- Chạy zero-shot: model TORGO (đóng băng) + rules Vosk WPM → 2 bảng:
  (a) TORGO transfer sang VN; (b) rules VN nội sinh.
- Kỳ vọng: TORGO zero-shot sẽ không tốt như 0.992 → đây là PHÁT HIỆN
  domain-shift ngôn ngữ, kết luận "cần corpus dysarthria tiếng Việt" — trung thực.
- Lưu WAV + CSV per-sample vào `test_results/external_speech_vn_<ts>/`.

## PROTOCOL E3 — ARM TRÊN DỮ LIỆU THẬT ĐẦU TIÊN (A-01) — 18–20/09

- ≥15 người × 2 điều kiện trước camera: (1) giơ 2 tay ngang 20s; (2) giơ 2 tay
  rồi **tự thả rơi 1 tay** (mô phỏng drift NIHSS item5) 20s.
- Chạy `arm_module` (rules + ML synthetic) lần đầu trên data thật → sens/spec.
- Kết quả dù thấp cũng là **eval thật đầu tiên** — gỡ nhãn "synthetic, chưa eval".

## PROTOCOL SYS-18 — NIHH ANCHOR (50 video có NIHSS chuẩn) — 19–23/09

- Nguồn: video demo NIHSS công khai có điểm bác sĩ công bố (kênh stroke
  education), mỗi video chấm tay 4 items Golden-Watch đo được.
- 2 chấm viên độc lập → thống nhất trước khi so với máy.
- Tính: MAE, RMSE, R², **Bland–Altman**, weighted-κ (đúng BANG_THANG_DO mục 2).
- Nếu chỉ tìm được <50 video: chấm được bao nhiêu lấy bấy nhiêu, ghi n thật.

## PROTOCOL F — FIELD STUDY (đã có trong kế hoạch T4, chuẩn hóa lại) — 25/09–03/10

- 3 hộ gia đình × 3 ngày chạy `web_server.py` liên tục.
- Số cần: **báo động giả / 24h** (kỳ vọng <1), uptime %, số lần crash,
  độ trễ dấu hiệu→còi (kịch bản bấm giờ), ghi chú điều kiện thật.
- Đây là bằng chứng "prospective at home" — bậc 3 của bậc thang.

---

## FRAMING TRÌNH BÀY (đưa vào tóm tắt báo cáo + 3 câu trả lời phỏng vấn)

1. *"Chúng tôi không có bệnh nhân nên KHÔNG tuyên bố phát hiện lâm sàng.
   Thay vào đó chúng tôi hoàn thành đủ 3 bậc kiểm định mà TRIPOD+AI yêu cầu
   cho AI y tế — bậc external + prospective mà phần lớn mô hình đã công bố
   còn thiếu."*
2. *"2 đóng góp phương pháp: Giao thức BlockSweep (AUC 1.000 là chỉ số lỗi,
   chứng minh bằng quét block) và ablation định lượng từng lớp defense."*
3. *"Mọi con số kèm n, CI 95%, ngưỡng, protocol ghi trước khi chạy — trung
   thực ở chỗ ghi rõ cái gì CHƯA có (arm synthetic→nay đã E3, speech VN,
   NIHSS ground truth)."*

## TIMELINE NÉN (10/09 → 05/10)

| Ngày | Việc |
|---|---|
| 10–14/09 | Radar thật 10 kịch bản (M5-02) + thu E1 face external |
| 15–17/09 | Đánh giá E1 → AUC external + figure |
| 18–20/09 | E3 arm thật + bắt đầu SYS-18 (50 video) |
| 21–23/09 | **Protocol B 100 tình huống × defense ablation** → 2 figure |
| 24–28/09 | SYS-18 xong (Bland-Altman/wκ) + E2 speech VN |
| 25/09–03/10 | Protocol F field study (chạy song song ở 3 hộ) |
| 01–04/10 | Prototype vỏ + video <3 phút + poster + báo cáo 15 trang (nhét 3 bậc thang + 2 đóng góp vào mục 4) + PL1/2/3 |
| 05/10 | NỘP HỒ SƠ |

## NGUYÊN TẮC KHÔNG ĐỔI (SYS-15)

1. Model đóng băng khi external — không tune lại trên tập external.
2. Protocol ghi TRƯỚC khi thu (đăng ký trong sổ nhật ký).
3. Số external thấp hơn internal = kết quả khoa học, không phải thất bại.
4. Mỗi protocol lưu script + JSON + CSV + PNG (seed 42, tái lập được).
