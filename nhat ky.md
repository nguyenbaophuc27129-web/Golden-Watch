# NHẬT KÝ THAY ĐỔI — Golden_Watch_KHKT (do trợ lý AI thực hiện, người dùng theo dõi)

> Mỗi thay đổi code/kết quả kiểm định đều ghi vào đây. Người dùng ghi chép lại
> vào SỔ NHẬT KÝ NGHIÊN CỨU chính thức (Phụ lục 2 hồ sơ dự thi) theo quy định KH.
> Quy ước ID: NK-xx (nhật ký nâng cấp). Các entry cách nhau bằng đường kẻ.

---

## NK-01 — 10/09/2026 · Figure "Giao thức BlockSweep" (đóng gói phát hiện leakage)

- **Việc:** tạo `training/plot_blocksweep.py`; render figure chuẩn khoa học từ
  `test_results/leak_check_blocksize.json` (số thật đã có từ 09/09).
- **Kết quả:** `test_results/blocksweep_leakage_diagnostic.png` — LogReg
  0.9419→0.9414→0.9348→0.9081 (block 50→1000, ổn định = tổng quát thật) vs
  HistGB 0.9993→0.9996→0.9979→0.9989 (≈1.000 mọi cỡ block = memorize người).
- **Ý nghĩa:** biến phát hiện SYS-28 thành 1 figure có tên gọi chính thức
  "Giao thức BlockSweep" để đưa vào báo cáo + trả lời chất vấn "vì sao không
  công bố số 100%?".
- **Kiểm định:** số lấy trực tiếp từ JSON, script chạy PASS, không sửa code hệ thống.

---

## NK-02 — 10/09/2026 · Master plan nâng tầm + Defense ablation harness

- **Việc:** (1) tạo `KE_HOACH_NANG_TAM_KHOA_HOC.md` (v1.0, cập nhật v2.0 sau
  góp ý của người dùng: KHÔNG thu dữ liệu người ngoài, KHÔNG 3 hộ gia đình);
  (2) tạo `training/protocol_b_defense_ablation.py` — 2 lệnh `capture` (camera
  thật 2s/chu kỳ, gắn nhãn kịch bản bằng phím) và `eval` (replay JSONL qua
  FusionEngine + DefenseEngine THẬT ×4 cấu hình off / L1+L4 / +L2 / full+L3,
  L3 dùng clock ảo 2s/chu kỳ, L1 calibrate từ đoạn binh_thuong).
- **Kiểm định:** smoke-test bằng dữ liệu tổng hợp (đã xóa khỏi test_results
  để không nhiễm thư mục bằng chứng): off = 18.33% FPR → full(+L3) = 0% —
  harness phân biệt được các lớp. **Phát hiện thiết kế:** L2 context chỉ gắn
  `defense_note` KHÔNG đổi điểm (đúng thiết kế fix L-08 chống che đột quỵ
  thật) → khi chạy thật, giảm FPR chủ yếu nhờ L3.
- **Việc cần người:** chạy capture thật với 100 tình huống (Protocol B,
  21–23/09) rồi `eval` → số FPR thật + figure.

---

## NK-03 — 10/09/2026 · S-1: PHÁT HIỆN LEAKAGE THỨ 2 — SPEECH 0.992 LÀ SESSION-LEVEL, ĐÁNH GIÁ LẠI THEO NGƯỜI THẬT (LOSO)

- **Phát hiện:** `train_speech_torgo.py` chia train/test theo `hash(speaker_dir)`
  mà speaker_dir = **session×mic** (`wav_arrayMic_F03S01` ≠
  `wav_headMic_F03S01`) → cùng 1 người thật (VD F03) nằm ở CẢ train lẫn test
  (khác mic). Số AUC 0.992 chưa phải người-level — giống đúng kịch bản face
  HistGB (SYS-28).
- **Việc:** viết `training/eval_speech_speaker_loso.py` — LOSO theo 15 NGƯỜI
  THẬT (strip prefix mic + hậu tố session: F01/F03/FC01/M04…), 2 model (LogReg
  đầu tuyến tính + MLP cùng kiến trúc production 48→256/128/64), subsample ≤20
  file/session (seed 42, protocol ghi TRƯỚC), 1,100 file dùng được.
- **Kết quả chính thức** `test_results/speech_speaker_loso_20260910_214834/`:
  - **P1 LOSO: LogReg AUC 0.620** (thr 0.661, Sens 51.1 [46.9–55.3], Spec 75.2
    [71.4–78.6]) · **MLP AUC 0.609** (Sens 48.7 / Spec 75.5) — so với 0.992
    session-level: **giảm ~0.37 AUC = leakage thứ 2 được chứng minh bằng số**.
  - P2 per-mic OOF: arrayMic 0.562 · headMic 0.669.
  - P3 cross-mic (người ngoài fold): array→head 0.592 (LogReg) ·
    **head→array 0.415** (dưới 0.5!) → model KHÔNG bền với đổi micro.
- **Ý nghĩa:** (1) sửa báo cáo — KHÔNG công bố 0.992 như người-level; số đúng
  là ~0.62 người-level (vẫn có tín hiệu nhưng yếu); (2) đóng góp phương pháp
  thứ 2 cạnh BlockSweep: "tự phát hiện 2 case leakage bằng protocol đúng, không
  chờ giám khảo"; (3) hướng cải thiện thật: S-2 openSMILE/wav2vec2 + ghi
  limitation người-level trong báo cáo.
- **Kiểm định:** script chạy PASS, protocol seed 42 ghi trước, output CSV OOF +
  JSON + 3 PNG (roc/per-speaker/mic) lưu đủ để tái lập.

---

## NK-04 — 10/09/2026 · F-1a: FACE v3 MULTI-SEED (10 SEEDS) — AUC 0.943 ỔN ĐỊNH, CÔNG BỐ ĐƯỢC DẠNG KHOẢNG

- **Việc:** tạo `training/face_v3_multiseed_stability.py` — trích đặc trưng
  MediaPipe 1 LẦN (cache `_face_v3_features_cache.npz`, 3,715 ảnh × 28 ft),
  lặp protocol v3 G4_all với 10 seed (42–51) chỉ random lại phép chia block.
  KHÔNG đụng artifact models/.
- **Kết quả** `test_results/face_v3_multiseed_20260910_215359/`:
  **AUC test mean 0.9398 ± 0.0099 [0.9282–0.9557]** · sens mean 0.8164 ·
  spec mean 0.9172; SD chỉ ~0.01 → mô hình tuyến tính v3 THẬT SỰ ổn định
  (đối lập hoàn toàn với HistGB ≈1.000 mọi seed = leakage đã chứng minh ở
  BlockSweep/NK-01).
- **Ý nghĩa:** trả lời chất vấn "chia khác thì sao?" bằng số: công bố
  **0.94 ± 0.01 (10 seed)** thay vì 1 số 0.943 duy nhất — chuẩn mực quốc tế
  (TRIPOD+AI khuyến nghị phân phối thay vì điểm 1 lần chia).
- **Kiểm định:** 10/10 seed chạy PASS ~10 phút, figure
  `multiseed_stability.png` (đường AUC theo seed + băng ±SD), summary.json
  đủ 10 runs để tái lập.

---

## NK-05 — 10/09/2026 · SYS-18: NIHSS VIDEO TOOLKIT (template + score-video + eval)

- **Việc:** tạo `training/nihss_video_toolkit.py` — bộ công cụ hoàn chỉnh cho
  Protocol SYS-18 (50 video NIHSS có điểm bác sĩ công bố):
  1. `template` → `NIHSS_SY18_bang_cham_video.csv` (2 chấm viên độc lập
     rater1/rater2 + consensus agreed + ref_total công bố).
  2. `score-video <file.mp4>` → chạy 4 module THẬT (face ML v3, arm YOLO,
     gait YOLO, speech MLP production) trên từng video, sample 2 frame/s,
     prob gộp theo median → `estimate_nihss` → điền cột machine_* vào CSV.
     Audio tách bằng ffmpeg đóng gói trong `imageio-ffmpeg` (không cần cài
     ffmpeg hệ thống).
  3. `eval` → MAE, RMSE, r², **weighted-κ quadratic** từng item + TOTAL,
     **Bland–Altman** (bias + LoA 1.96SD), inter-rater κ, 3 figure
     (bland_altman_total / scatter_total / per_item_stats) + summary.json.
- **Kiểm định:** py_compile PASS; smoke eval dữ liệu giả PASS đầy đủ số +
  figure (đÃ XÓA thư mục output smoke khỏi test_results để không nhiễm
  bằng chứng); template tạo PASS.
- **Việc cần người:** tải 50 video → `score-video` → 2 chấm viên điền
  rater1/rater2/agreed → chạy `eval` → số so máy-người chính thức.

---

## NK-06 — 10/09/2026 · S-2: openSMILE eGeMAPSv02 (88 ft) vs 48 ft tự trích — CÙNG FILE, CÙNG FOLD

- **Việc:** tạo `training/eval_speech_opensmile_loso.py` — trích 88 features
  eGeMAPSv02 (chuẩn quốc tế trong tài liệu rối loạn nói) bằng openSMILE 2.6.0,
  đánh giá LOSO **trên đúng 1,100 file và đúng fold người** của lần chạy gốc
  (nạp danh sách từ `oof_predictions.csv` của NK-03) → so sánh công bằng
  apples-to-apples, LogReg đầu tuyến tính.
- **Kết quả** `test_results/speech_opensmile_loso_20260910_221102/`:
  - eGeMAPSv02 88 ft: **AUC 0.663** (thr 0.659, Sens 58.0 [53.8–62.1],
    Spec 75.4 [71.6–78.7]).
  - 48 ft tự trích (production): AUC 0.620 → **Δ = +0.043**.
- **Quyết định trung thực:** +0.043 là cải thiện NHỎ → production GIỮ 48 ft
  (đơn giản, không phụ thuộc thư viện ngoài, chạy on-device); so sánh 2 bộ
  features được công bố như bằng chứng nghiên cứu đặc trưng có kiểm soát.
  Đồng thời khẳng định: speech người-level vẫn là module YẾU nhất (~0.62–0.66)
  → limitation rõ trong báo cáo, hướng tương lai wav2vec2 + corpus tiếng Việt.
- **Kiểm định:** 1,100/1,100 file trích opensmile 0 lỗi, 0 ô NaN; cùng seed 42;
  summary.json + oof_opensmile.csv + figure so sánh AUC đã lưu.

---

## NK-07 — 10/09/2026 · HỒI QUY CUỐI NGÀY SAU TOÀN BỘ NÂNG CẤP

- **Việc:** chạy lại 2 bộ test chuẩn sau khi thêm 5 script mới
  (multiseed, LOSO, opensmile, NIHSS toolkit, plot BlockSweep):
  - `src/test_sys14b_regression.py`: **10/10 PASS**.
  - `src/test_all_metrics.py`: **58/58 PASS**.
- **Ý nghĩa:** mọi nâng cấp hôm nay đều là THÊM script đánh giá/figure, KHÔNG
  đụng code hệ thống (src/, web_server.py) → pipeline sản phẩm không đổi,
  an toàn demo.

---

## NK-08 — 11/09/2026 · BỘ CÔNG CỤ E2/E3 + PL1 KHAI BÁO AI + BỘ CÂU HỎI PHỎNG VẤN (chuẩn bị deadline 13/09)

- **Việc:**
  1. `training/eval_arm_e3.py` — eval arm trên VIDEO THẬT: 2 thư mục
     `thu_tuc\*.mp4` (label 0) vs `tha_tay\*.mp4` (label 1), chạy
     ArmWeaknessDetector thật 2 fps, gộp median video → AUC + Youden +
     sens/spec Wilson + confusion + figure. Ghi rõ "smoke test n nhỏ".
  2. `training/eval_speech_vn_e2.py` — E2 speech tiếng Việt ZERO-SHOT: 2 thư
     mục `vn_thuong\*.wav` vs `vn_liu\*.wav`, chạy SpeechAnalysisModule
     production (đóng băng), nạp mp3/m4a qua imageio-ffmpeg → AUC + accuracy
     + figure. Kỳ vọng trung thực: thấp hơn = phát hiện domain-shift.
  3. `PL1_khai_bao_su_dung_AI.md` — khai báo AI TRUNG THỰC theo đúng quy chế
     (bảng công cụ, AI làm gì / học sinh làm gì, biện pháp kiểm chứng qua
     sổ nhật ký, chỗ ký).
  4. `HOI_DA_PHONG_VAN.md` — 30 câu phỏng vấn khó nhất + câu trả lời theo số
     thật (BlockSweep, leakage 0.992→0.62, LogReg vs HistGB, defense 4 lớp,
     NIHSS 13 điểm, hạn chế, AI declaration).
- **Kiểm định:** py_compile PASS cả 2 script; PL1/Q&A là tài liệu.
- **Việc cần người:** quay video E3 + ghi âm E2 + tải video NIHSS + cắm radar
  → chạy các script → kết quả ghi NK-09 (E3), NK-10 (E2).

---
