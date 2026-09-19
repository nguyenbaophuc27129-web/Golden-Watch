# SỔ NHẬT KÝ NGHIÊN CỨU — GOLDEN-WATCH (PSCS)

> **File chính thức từ 12/09/2026** — mọi thay đổi code/kết quả kiểm định đều
> ghi vào đây (chuyển tiếp từ `nhat ky.md`, giữ nguyên thứ tự NK-xx).
> Người dùng chép vào SỔ NHẬT KÝ in theo Phụ lục 2 hồ sơ dự thi.
> Quy ước ID: NK-xx. Các entry cách nhau bằng đường kẻ.

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
  → chạy các script → kết quả ghi NK tiếp theo.

---

## NK-09 — 12/09/2026 · CHỐT HƯỚNG NÂNG CẤP v3.0 "HỆ GIÁM SÁT TỰ TIN" (A+B+D) + chuyển sang sổ nhật ký chính thức

- **Bối cảnh:** mục tiêu giải Nhì quốc gia; web search quota hết (reset
  13/09 22:22) → tính mới sẽ đối chiếu lại literature sau. Đã đối chiếu
  4 hướng candidate, người dùng chọn gói A+B+D (khuyến nghị).
- **Quyết định:** thêm v3.0 vào `KE_HOACH_NANG_TAM_KHOA_HOC.md` — 3 trụ cột,
  MỌI thứ là LỚP BỔ SUNG qua flag (fallback = hệ thống cũ đã eval), model
  đóng băng, seed 42:
  - **A — Uncertainty + Abstention ("biết mình không biết"):** Platt-calibrate
    prob từng module trên OOF ĐÃ LƯU (speech oof_predictions.csv; face tính lại
    từ cache `_face_v3_features_cache.npz`; gait LOSO chạy lại — 15 subject);
    uncertainty = spread giữa cửa sổ; luật abstain → trạng thái `NEEDS_CHECK`
    thay vì báo giả. Đo risk–coverage, selective-AUC, FPR@TPR cố định.
    File mới: `src/fusion/uncertainty.py` + `training/eval_abstention.py` →
    figure `abstention_curve.png`.
  - **B — Temporal Early-Warning (nâng cấp L3):** từ luật cửa sổ 30s thành
    **EWMA + CUSUM change-point + slope suy giảm** trên chuỗi fused score →
    trạng thái `DETERIORATING` riêng (suy giảm tiến triển ≠ emergency đơn điểm).
    Đo bằng replay harness FakeClock (độ trễ + FPR) + giả lập suy giảm trên
    log 72h + Protocol F. File mới: `src/defense/early_warning.py` +
    `training/eval_early_warning.py` → figure `early_warning_trajectory.png`.
    Cơ sở có sẵn: DefenseEngine L3 `_trend()` (defense_engine.py:207).
  - **D — Cross-modal sensor-check (radar↔camera):** đồng thuận chuyển động
    cửa sổ 2s; lệch → `defense_note` + giảm độ tin cậy hiển thị, GIỮ NGUYÊN
    điểm (bài học L-08). Demo: che camera → radar vẫn bắt ngã. File mới:
    `src/defense/cross_check.py`.
- **Timeline v3.0:** 12–14/09 A · 15–17/09 B · 18–20/09 D + radar thật + quay
  E3 · 21–23/09 Protocol B × ablation (thêm 2 cấu hình abstain/cross-check) ·
  24–28/09 SYS-18 + E2 · 25/09–03/10 Protocol F 72h · 01–04/10 hồ sơ ·
  05/10 NỘP.
- **Quy trình mới:** từ nay MỌI thay đổi ghi vào `so_nhat_ky.md` (file này —
  sổ nhật ký chính thức, chuyển tiếp từ `nhat ky.md`).
- **Kiểm định:** chưa có code mới hôm nay — chỉ tài liệu kế hoạch; các script
  A/B/D sẽ có entry kiểm định riêng khi chạy.

---

## NK-10 — 12/09/2026 · TRỤ CỘT A: UNCERTAINTY + ABSTENTION — FACE "BIẾT MÌNH KHÔNG BIẾT" (0.777), SPEECH THẲNG THẮN THỪA (0.488 ≈ RANDOM)

- **Việc:**
  1. `src/fusion/uncertainty.py` — thư viện thuần numpy: Platt calibration
     (Newton 1D + phiên bản CV out-of-fold), ensemble mean/std, margin,
     combined uncertainty, abstain top-frac, risk–coverage + selective summary.
  2. `training/eval_abstention.py` — face: OOF GroupKFold-5 theo block × 10
     seed (42–51) từ cache npz (khác NK-04 80/20 vì cần OOF per-image — khai
     báo trong JSON); speech: tái dùng đúng OOF CSV NK-03 (1,100 file,
     LOSO người thật); threshold Youden từng module.
  3. FusionEngine cắm flag `abstain=False` (mặc định TẮT): module có thể
     mang `uncertainty`; gộp có trọng số → `needs_check` khi > 0.6.
     **EMERGENCY KHÔNG BAO GIỜ bị hạ cấp** (an toàn trước, đúng L-08).
- **Kết quả chính thức** `test_results/abstention_20260912_122517/`:
  - **FACE — bất định có giá trị:** AUC ensemble 0.9413 (platt-CV 0.9393 —
    AUC không đổi, đúng tính chất đơn điệu) · **unc_error_auc 0.777**
    (bất định dự đoán được lỗi). **Abstain 20% bất định nhất → err
    11.66→6.56% (−5.1đ), FPR 9.5→5.3%, sens thậm chí TĂNG 91.1%**; lỗi
    trong phần bị bỏ 32% ≈ 2.7× đậm đặc lỗi trung bình → bỏ ĐÚNG CHỖ.
  - **SPEECH — phát hiện trung thực:** ensemble 0.6219 (LogReg 0.6202 ·
    MLP 0.6093) · **unc_error_auc 0.488 ≈ random** — bất đồng 2 model
    KHÔNG dự đoán được lỗi → model yếu sai "TỰ TIN". Abstain 20% chỉ giảm
    0.75đ err. Kết luận: layer bất định còn là THƯỚC ĐO ĐỘ TIN CẬY module
    (face đáng tin ở vùng xám, speech thì không) — feed trực tiếp vào
    câu chuyện fusion + limitation speech trong báo cáo.
- **Ý nghĩa:** (1) figure `abstention_curves.png` + `uncertainty_separation.png`
  = bằng chứng "hệ biết mình không biết" cho chất vấn; (2) số FPR 5.3% @
  sens 91.1% (face, abstain 20%) chạm đúng mục tiêu Protocol B (EMERGENCY
  giả ≤5%) ở cấp module; (3) speech 0.488 là limitation được ĐO, không bịa.
- **Kiểm định:** unit test flag 4/4 PASS (default off giống hệ cũ · ON
  needs_check đúng · EMERGENCY intact · uncertainty thấp im lặng);
  regression sau khi đụng src/fusion: **10/10 + 58/58 PASS**; smoke 2-seed
  đã xóa khỏi test_results; figure + JSON + CSV per-sample lưu đủ tái lập.

---

## NK-11 — 12/09/2026 · TRỤ CỘT B: EARLY WARNING (EWMA+CUSUM) — BẮT ĐƯỢC SUY GIẢM CHẬM DƯỚI NGƯỜNG MÀ LUẬT CŨ KHÔNG BAO GIỜ BÁO (0/20 vs 20/20), NHANH HƠN ~50% Ở MỌI KỊCH BẢN

- **Việc:**
  1. `src/defense/early_warning.py` — EarlyWarningMonitor thuần numpy,
     clock injectable (test được FakeClock): EWMA alpha 0.15 + CUSUM một
     phía (k=0.5σ, h=5σ chuẩn hóa theo baseline cá nhân mu0/σ từ đoạn
     bình thường, median+MAD) + xác nhận run-length 15 chu kỳ cao liên
     tiếp (30s — nhất quán PERSISTENT_SECONDS) + slope OLS 5 phút.
     Trạng thái: OK / WATCH (EWMA ≥ mu0+2σ) / DETERIORATING (CUSUM>h +
     run 30s) — suy giảm tiến triển ≠ emergency đơn điểm.
  2. DefenseEngine cắm hook `early_warning=None` (mặc định KHÔNG có =
     hệ cũ nguyên vẹn): verdict() cho monitor xem điểm; DETERIORATING →
     alert=True reason riêng. Fast-path EMERGENCY KHÔNG đổi.
  3. `training/eval_early_warning.py` — so OLD (DefenseEngine THẬT +
     FakeClock, cùng pattern harness) vs NEW trên dữ liệu MÔ PHỎNG
     (ghi rõ simulated, seed 42, chu kỳ 2s, warm-up 200 chu kỳ): nền
     AR(1) người khỏe + spike biểu cảm + burst stress + 4 ramps.
- **Bài học thiết kế khi smoke:** WATCH theo "CUSUM > h/2" làm WATCH
  nhấp nháy 5,905/24h sau mỗi spike (CUSUM còn cao hàng phút) → đổi
  thành "EWMA ≥ mu0+2σ" → 16/24h episode, khớp số spike. Đánh đổi ghi
  lại làm bằng chứng quy trình.
- **Kết quả chính thức** `test_results/early_warning_20260912_124947/`
  (FAR: 6h × 20 seed = 120h; latency: 20 seed/kịch bản):
  - **FAR người khỏe: OLD 0/24h · NEW DETERIORATING 0/24h** (WATCH mềm
    16.6/24h ≈ số spike biểu cảm) · burst stress 3–5 chu kỳ @75–85:
    cả hai 0/24h → không đánh mất khả năng chống spike của L3 cũ.
  - **Latency phát hiện (median):** ramp +40/15ph OLD 10.7 → NEW **5.9
    phút** (−45%) · +40/30ph 20.1 → **10.0** (−50%) · +40/60ph 39.9 →
    **20.3** (−49%).
  - **HEADLINE — ramp +18 điểm/30ph kết thúc ~40 (DƯỚI ngưỡng cảnh báo
    50): OLD 0/20 — KHÔNG BAO GIỜ báo · NEW 20/20 tại median 21.6 phút.**
    Đây đúng lỗ hổng "prodromal drift" của luật ngưỡng mà trụ cột B vá.
- **Ý nghĩa:** (1) figure `early_warning_trajectory.png` 4 panel + JSON
  = bằng chứng "coi DÒNG THỜI GIAN, không phải khoảnh khắc"; (2) trạng
  thái DETERIORATING mới cho UI: cảnh báo "đang xấu dần, người thân kiểm
  tra" — điểm cộng Trình bày; (3) khi có Protocol B thật (21–23/09)
  chạy lại replay thay simulated — protocol đã ghi sẵn trong JSON.
- **Kiểm định:** unit test monitor PASS (spike ngắn không báo · ramp
  dưới ngưỡng bắt được 21.9 phút); unit test hook 3/3 (default nguyên
  vẹn · DETERIORATING alert đúng · fast-path không đổi); regression sau
  khi đụng src/defense: **10/10 + 58/58 PASS**; smoke output đã xóa.

---

## NK-12 — 12/09/2026 · TRAIN 100% DATASET CHO MODULE CÓ DATA + TÀI LIỆU TỔNG KẾT NGHIÊN CỨU

- **Nguyên tắc trung thực (TRIPOD+AI) ghi trước:** số ĐÁNH GIÁ công bố trong
  báo cáo VẪN là protocol đã khóa (face 0.943 / 0.94±0.01 multi-seed; speech
  LOSO 0.62; gait LOSO 0.879). Mô hình fit-100% là ARTIFACT PRODUCTION/NGHIÊN
  CỨU sau khi validation khóa — chuẩn "final fit after locked validation".
  KHÔNG bao giờ công bố số fit-100%.
- **Việc 1 — Face 100%:** `training/train_face_v3_full.py` refit LogReg
  trên 100% 3,715 ảnh (C chọn GroupKFold OOF nội bộ → C=0.03, OOF all-data
  0.9419; threshold Youden 0.298→0.38). **Artifact mới
  `models/face_blend_v3_full_20260912_181937.json`** — coef tương quan 0.9929
  với bản cũ (swap an toàn), `face_ml_v3.load_latest` tự nạp bản full
  (verify PASS). Artifact giữ nguyên số khóa (auc_test/sens/spec từ SYS-28 +
  multiseed NK-04).
- **Việc 2 — Speech 100% TORGO:** `training/train_speech_torgo_full.py`
  trích 48-ft TOÀN BỘ dataset: **17,631/17,633 file thành công** (2 lỗi,
  6,176 dys / 11,455 con, cache resumable `_speech_features_full.npz`,
  ~32 file/s — nhanh hơn ước tính 20×) → train MLP cuối [256,128,64]
  seed 42 epoch 150 (loss cuối ~0.13). **Artifact
  `models/speech_torgo_full_20260912_183425.pth`** — smoke hướng dự đoán
  ĐÚNG cả 2 lớp (con → prob dys 0.033 · dys → 0.998); fit-accuracy trên
  chính tập train 0.982 (**chỉ tham chiếu — KHÔNG công bố**, số chính thức
  vẫn LOSO người thật 0.62). **Production KHÔNG đổi** (module2_main
  hard-code model cũ + ngưỡng 30/56% đã chốt từ extended test M2-10/12 —
  muốn swap phải chạy lại extended test). JSON metadata ghi rõ mô hình
  nghiên cứu + ref LOSO/openSMILE.
- **Việc 3 — Gait:** KHÔNG retrain — model v2 PhysioNet đã train đủ và
  KHÔNG nối camera theo quyết định M4-07 (sai khác không gian đặc trưng
  thảm lực vs pose camera). Camera gait giữ rule-based.
- **Việc 4 — Tài liệu:** tạo `TONG_KET_NGHIEN_CUU.md` (v1.0) — tổng kết
  sản phẩm 3 câu, sơ đồ nguyên lý ASCII 5 module→defense→fusion→triage→
  handoff, sơ đồ thuật toán face/early-warning/abstention/speech, BOM
  ≈1,010,000đ, bảng 10 thang đo khoa học (Wilson/Youden/LOSO/BlockKFold/
  multi-seed/Bland–Altman/wκ/DeLong/FAR/risk–coverage), tự chấm theo KH
  (≈81.5/100), định vị mới "3 lớp chưa từng có" + câu chốt hội đồng,
  bảng việc còn lại tới 05/10.
- **Ý nghĩa:** mọi dataset có sẵn (face Kaggle, speech TORGO, gait
  PhysioNet) giờ được khai thác 100%; chỉ còn E2/E3/NIHSS/Protocol B/radar
  là dữ liệu ngoài cần người thu (đã có checklist chuẩn bị).
- **Kiểm định:** face verify loader PASS + regression sau khi swap:
  **10/10 + 58/58 PASS**; speech job chạy nền (cache tăng đều); artifact
  trùng từ lần chạy lỗi đã xóa.

---

## NK-13 — 12/09/2026 · CHECKLIST CHUẨN BỊ DỮ LIỆU CHO ĐỘI + QUY ĐỊNH SỐ NK CHO CÁC EVAL SẮP TỚI

- **Việc 1 — Tạo `CHUAN_BI_CHECKLIST.md`** (root, in được): hướng dẫn đội thu
  dữ liệu ngoài — §0 GVHD ký kế hoạch (KHẨN, không ký = không chấm) + hạnh kiểm;
  §1 mua sắm ≈1,010,000đ (LD2450 baud 256000, C270, vỏ, cáp, đế); §2 quay video
  E3 tay (`thu_tuc/` giữ 2 tay 20s · `tha_tay/` thả 1 tay, 3–5 người × 2 clip
  ~30s, tên `ten_01.mp4`); §3 nối radar + 10 kịch bản M5-02; §4 ghi âm E2
  tiếng Việt 3 câu chuẩn (`vn_thuong/` · `vn_liu/`, ≥10 người, tên kèm tuổi);
  §5 SYS-18 NIHSS 50 video + 2 chấm viên vào CSV; §6 Protocol B 100 tình huống
  (25 cười/15 ngáp/15 quay nghiêng/15 ánh sáng yếu/15 khẩu trang/15 đi chậm);
  §7 72h liên tục; §8 hồ sơ dự thi.
- **Quy ước số NK cho eval khi đội báo "xong":** E3 → NK-14 · radar → NK-15 ·
  E2 → NK-16 · NIHSS → NK-17 · Protocol B → NK-18 · 72h → NK-19.
  Trụ cột D (`cross_check.py`, mới thăm dò code chưa viết) khi làm tiếp sẽ
  nhận số NK trống tiếp theo — không đặt chỗ trước.
- **Quy định mới của user (áp dụng từ entry này):** MỌI việc dù nhỏ nhất cũng
  phải ghi vào sổ nhật ký này.
- **Kiểm định:** không đổi code/model — chỉ tài liệu hướng dẫn.

---

## NK-14 — 12/09/2026 · VIẾT NHẬT KÝ DỰ ÁN THEO NGÀY 01/06→12/09 (`NHAT_KY_DU_AN.md`)

- **Việc:** khai thác 2 thư mục lịch sử `C:\Users\Admin\Documents\KHKT_2026`
  (giai đoạn ý tưởng + thủ tục bệnh viện) và `C:\Users\Admin\Documents\NCKHKT_26`
  (fga_project + root docs) bằng 3 agent thăm dò song song; dựng dòng thời gian
  từ mtime file + timestamp trong tên artifact + ngày viết trong tài liệu +
  git log (trung thực: ngày không có bằng chứng ghi rõ "không còn dấu vết",
  mtime 06/09 19:52 là lúc batch copy không phải ngày viết code).
- **Tạo `NHAT_KY_DU_AN.md`** (root) — bản gốc sổ nhật ký PL2: tóm tắt 6 lần
  đổi ý tưởng lớn (gait+C# 03/2026 → rPPG/RAG 14/05 → Stroke Detect System +
  bệnh viện 19–22/06 → FAST đa cảm biến + radar 06/07 → FGA v3.0 28/07 →
  Golden Watch 19/08 sau góp ý phê phán → ML thật 25/08+ → tự bắt leakage
  07–10/09 → "Hệ giám sát tự tin" 12/09); nhật ký từng ngày 01/06→12/09
  (tháng 6 hoạch định + đơn BV, tháng 7 chuyển trục FAST + bộ mẫu đơn 15/07,
  tháng 8 code hóa + crisis 30/08, tháng 9 kiểm định + trung thực hóa số);
  bảng "số quá đẹp đã tự hủy" 5 dòng (93.75% no-split, gait 96.88% window,
  speech 0.992 session, arm/visual synthetic, HistGB 1.000); phần việc chờ
  dữ liệu.
- **Đổi quy ước số NK (supersede NK-13):** do NK-14 đã dùng cho file này →
  E3→**NK-15** · radar→**NK-16** · E2→**NK-17** · NIHSS→**NK-18** ·
  Protocol B→**NK-19** · 72h→**NK-20**. Đã đồng bộ `CHUAN_BI_CHECKLIST.md`
  + `NHAT_KY_DU_AN.md`.
- **Bối cảnh:** user đi thu dữ liệu (E2/E3/radar); đề nghị viết nhật ký
  từng ngày từ 01/06 để làm sổ PL2.
- **Kiểm định:** không đổi code/model — chỉ tài liệu.

---

## NK-15 — 12/09/2026 · HƯỚNG DẪN CHUẨN BỊ DỮ LIỆU + TẠM DỪNG TRỤ CỘT D

- **Việc:** user yêu cầu "hướng dẫn chuẩn bị dữ liệu" → giao trực tiếp 5 mục
  cho đội (nội dung trùng `CHUAN_BI_CHECKLIST.md` NK-13): E3 arm video /
  E2 ghi âm tiếng Việt / SYS-18 NIHSS / radar LD2450 10 kịch bản / Protocol B
  100 tình huống; kèm mapping NK sẽ ghi khi đội báo "xong".
- **Trụ cột D (`cross_check.py`):** thăm dò kiến trúc xong (hooks
  `defense_engine.py`: `early_warning`, `defense_note`; `web_server.py`
  analysis_worker + radar metrics) — TẠM DỪNG theo ưu tiên của user;
  task #13–15 giữ pending, khi làm tiếp nhận số NK trống tiếp theo.
- **Kiểm định:** không đổi code/model — chỉ hướng dẫn.

---

## NK-16 — 12/09/2026 · VIẾT LẠI `NHAT_KY_DU_AN.md` THEO MẪU PHỤ LỤC 2 (6 mục/ngày)

- **Việc:** user yêu cầu điều chỉnh nhật ký theo mẫu chính thức
  `6756phuluc2-hd-so-nhat-ky-nghien-cuu_127202612.docx`, mỗi ngày dài
  1–3 trang. Trích template từ docx (python zipfile + regex
  word/document.xml): mỗi ngày NGÀY/TRANG · GIAI ĐOẠN · THỜI GIAN + 6 mục
  (Mục tiêu / Dụng cụ & Vật liệu / Tiến trình & Hiện tượng / Kết quả & Số
  liệu thô / Rút kinh nghiệm & Lỗi sai / Kế hoạch tiếp theo) + dòng chữ ký
  HS / GV-GMC; bìa + mục lục để trống 2–3 trang + phụ lục cuối sổ.
- **Kết quả:** viết lại XONG 104 ngày (01/06→12/09), mỗi ngày đủ 6 mục +
  chữ ký; ngày có bằng chứng file lấy THỜI GIAN theo mtime thực, ngày tái
  dựng đánh dấu *[r]*; giữ nguyên số thô sai/ảo kèm ghi chú "SAU NÀY NHẬN RA".
  Cuối file 3 phụ lục: A — bảng 6 lần đổi ý tưởng; B — bảng 5 số "quá đẹp
  đã tự hủy"; C — phương pháp dựng nhật ký + cảnh báo mtime 19:52 ngày
  06/09 (batch copy) + hướng dẫn chép sang sổ giấy PL2.
- **Đổi quy ước số NK (supersede NK-14):** do NK-15/NK-16 đã dùng →
  E3→**NK-17** · radar→**NK-18** · E2→**NK-19** · NIHSS→**NK-20** ·
  Protocol B→**NK-21** · 72h→**NK-22**. Đã đồng bộ `CHUAN_BI_CHECKLIST.md`.
- **Kiểm định:** không đổi code/model — chỉ tài liệu.

---

## NK-17 — 13/09/2026 · NÂNG CẤP NHẬT KÝ DỰ ÁN LÊN V2 (BIỂU ĐỒ + SỐ THẬT) · SỰ CỐ & KHÔI PHỤC V1

- **Việc:** user yêu cầu "viết chi tiết hơn, mỗi ngày ~3 trang, đưa số liệu
  thật cho các ngày kiểm thử, vẽ biểu đồ, chỉnh cho quyển sổ đọc nhất vô nhí".
- **Pipeline biểu đồ:** `training/plot_nhat_ky_charts.py` (matplotlib Agg,
  PYTHONUTF8=1, dpi 130) → `test_results/nhat_ky_charts/*.png` — 10 biểu đồ,
  100% số liệu parse TỰ ĐỘNG từ file kết quả thật (metrics_pack, multiseed,
  BlockSweep, speech LOSO, abstention, early-warning, log train speech —
  KHÔNG số nào gõ tay). Chạy lại: **OK 10/10**. Fix trước đó: chart 08 màu
  theo cột đúng; chart 01/10 legend dưới trục + nhãn mốc xen kẽ trên/dưới.
- **`NHAT_KY_DU_AN.md` v2:** viết lại toàn bộ — 2.366 dòng / 93 ngày
  (01/06→12/09), mỗi ngày đủ 6 mục Phụ lục 2 + chữ ký; bảng số liệu thật ở
  mọi ngày kiểm thử (07/09 metrics_pack: face rules n=478 acc 57.11% AUC
  0.638, face ML 0.845, gait LOSO n=162 AUC 0.879, speech n=55 0.992 ⚠️;
  BlockSweep 50/200/500/1000; 10-seed 0.9398±0.0099; speech LOSO 0.620;
  trụ A/B ngày 12/09); 10 biểu đồ nhúng đúng từng mốc; các khối
  **"※ Kẻ lại cho rõ"** (bình luận hồi tố đánh dấu riêng, KHÔNG sửa sự kiện
  — quy định tại Phụ lục C); **Phụ lục D mới:** atlas 10 biểu đồ + nguồn
  số liệu + lệnh chạy lại (truy vết được từng con số).
- **Sự cố & khôi phục (ghi minh bạch):** khi viết chunk 1, bản v1 bị GHI ĐÈ
  trước khi kịp đọc hết nội dung còn lại → khôi phục bằng cách replay 12
  thao tác Write/Edit của v1 từ transcript phiên làm việc (file .jsonl);
  xử lý trùng lặp do 1 op Edit từng FAIL cũng nằm trong transcript → tách
  khối 06/09–12/09 bị nhân đôi. Kết quả: `NHAT_KY_DU_AN_v1_recovered.md`
  (2.108 dòng, 93 ngày, 0 trùng) — GIỮ làm bản lưu. Kiểm tra cuối v2:
  0 anchor dư · 0 ký tự CJK · 10/10 ảnh tồn tại · 93 chữ ký.
- **Đổi quy ước số NK (supersede NK-16):** do NK-17 đã dùng cho entry này →
  E3→**NK-18** · radar→**NK-19** · E2→**NK-20** · NIHSS→**NK-21** ·
  Protocol B→**NK-22** · 72h→**NK-23**. Đã đồng bộ `CHUAN_BI_CHECKLIST.md`.
- **Kiểm định:** không đổi code/model — chỉ tài liệu + chạy lại script vẽ
  biểu đồ (output mới cùng đường dẫn cũ).

---

## NK-24 — 16/09/2026 · KIỂM ĐỊNH THỐNG KÊ BỔ SUNG: McNEMAR · DeLONG · CALIBRATION · PR-AUC (điền các ô "Kế hoạch/THIẾU" trong BANG_THANG_DO_KIEM_DINH.md)

- **Việc:** tạo `training/eval_statistical_tests.py` — đo TRÊN DỮ LIỆU THẬT
  có sẵn, KHÔNG đụng `models/`, KHÔNG đụng code hệ thống:
  face = cache `_face_v3_features_cache.npz` (3,715 ảnh × 28 ft), so 2 model
  trên CÙNG OOF fold GroupKFold(5) theo block (C chọn inner-OOF như protocol
  v3, seed 42); speech = dùng ĐÚNG `oof_predictions.csv` của NK-03
  (1,100 file LOSO người thật) — không chạy lại model nào.
- **Kết quả chính thức** `test_results/stat_tests_20260916_213450/`
  (summary.json + 3 PNG chuẩn khoa học):
  - **McNemar chính xác (binomial)** — face 5-ratio (cũ) vs v3 28 ft tại
    ngưỡng Youden từng model: 130 vs 137 cặp đúng/sai lệch,
    **p = 0.714 → KHÔNG khác biệt về accuracy tại operating point**.
  - **DeLong paired** — face: AUC v3 **0.9419** vs 5-ratio **0.9276**,
    Δ = 0.0143 [CI 95% 0.0093–0.0192], **p = 1.5e-08 → v3 tốt hơn ĐÁNG KỂ về
    khả năng phân biệt**. Speech: LogReg 0.6202 vs MLP 0.6093,
    Δ = 0.0109 [−0.0063–0.0280], **p = 0.213 → KHÔNG khác biệt** (hợp lệ
    chọn LogReg làm model nghiên cứu — đơn giản hơn mà không thua).
  - **Calibration face v3 (OOF):** Brier raw 0.0783 · ECE **2.1%** — prob của
    LogReg v3 ĐÃ calibrate tốt sẵn; Platt nested (fit trong fold) cho
    Brier 0.0792 / ECE 2.5% → **KHÔNG cải thiện thêm** — phát hiện trung
    thực: prob v3 dùng trực tiếp được cho abstention (NK-10) mà không cần
    bước calibrate riêng.
  - **PR-AUC:** face v3 **0.9237** (baseline dương 0.3341 — dữ liệu lệch
    1:2) · speech LogReg AP **0.6572** (baseline 0.4909 — trên baseline
    nhưng chỉ xấp xỉ mức AUC, khớp story module yếu nhất hệ thống).
- **Ý nghĩa:** (1) điền xong các ô "Kế hoạch" trong bảng thang đo — McNemar
  và DeLong giờ là SỐ THẬT có figure; (2) câu trả lời chất vấn chuẩn:
  *"blendshapes cải thiện khẳ năng phân biệt (DeLong p<0.001) chứ không
  phải accuracy tại một ngưỡng — hai khái niệm khác nhau"*; (3) prob v3
  đã calibrate sẵn (ECE 2.1%) — trụ cột A không cần Platt riêng cho face;
  (4) LogReg giữ vị trí model chính ở cả face lẫn speech có kiểm định
  thống kê đi kèm.
- **Kiểm định:** regression sau khi thêm script (KHÔNG đụng src/, web_server,
  models): `src/test_sys14b_regression.py` **10/10 PASS** +
  `src/test_all_metrics.py` **58/58 PASS** (chạy 16/09); 3 bug script ghi
  minh bạch: DeLong dùng công thức chuẩn Var(Δ) = l·Cov(v01)·l/m +
  l·Cov(v10)·l/n (m ≠ n được), import `fusion.uncertainty` cần thêm
  `src/` vào sys.path.
- **Đồng thời:** rà soát tính mới (novelty) 16/09 — hạn mức web search HẾT
  (reset 13/10/2026) → kết quả sơ bộ dựa trên kiến thức nền, ghi rõ 8 truy
  vấn cần chạy lại sau 13/10: (a) multimodal contactless stroke monitoring
  home camera radar · (b) radar facial asymmetry stroke · (c) selective
  prediction stroke screening abstention · (d) uncertainty-aware
  prehospital stroke AI · (e) contactless early warning score radar camera
  home · (f) LD2450 fall detection · (g) ISEF stroke detection 2025–2026 ·
  (h) học sinh cảnh báo đột quỵ AI. Sơ bộ: KHÔNG thấy hệ trùng tổng thể;
  rủi ro trùng lớn nhất = FAST.AI (JAMA Neurol 2025, webcam mặt+tay) và
  radar ngã tiêu dùng (Aqara FP2/ESPHome-LD2450) → tuyên bố tính mới ở tầng
  HỆ GIÁM SÁT + 3 trụ cột tự tin + BlockSweep/ablation, KHÔNG tuyên bố
  "phát hiện méo mặt là mới" hay "LD2450 là sáng tạo".

---

## NK-25 — 16/09/2026 · APP CẢNH BÁO TRÊN ĐIỆN THOẠI (PWA QUA HOTSPOT — KHÔNG INTERNET) · WEBSOCKET /ws + TRANG /mobile

- **Quyết định kiến trúc:** chọn **web-app cài được (PWA)** thay vì APK gốc —
  (1) hoạt động QUA HOTSPOT laptop → giữ nguyên điểm bán "offline, dữ liệu
  không rời máy"; APK + FCM push bắt buộc internet (mâu thuẫn câu chuyện
  dự án) và rủi ro trễ trước 05/10; (2) "Thêm vào Màn hình chính" = icon
  app như native. APK/WebView ghi vào "hướng phát triển".
- **Việc (sửa `web_server.py` — thuần ADD-ON, không đổi pipeline chấm điểm):**
  1. **Cờ `--lan`:** mặc định bind 127.0.0.1 như cũ (an toàn); có cờ →
     bind 0.0.0.0 cho điện thoại trong mạng hotspot/wifi nhà.
  2. **Kênh sự kiện:** `push_mobile_event()` — thread phân tích đẩy
     `status` (nhịp tim mỗi chu kỳ 2s: risk/score/camera/radar) và `alert`
     (khi ALERTER triggered, kèm score + NIHSS); deque 100 sự kiện + seq.
  3. **WebSocket `/ws`:** snapshot đầy đủ đầu tiên → stream sự kiện mới
     (poll seq 0.5s/client — ≤3 điện thoại không đáng kể).
  4. **Trang `/mobile`** (PWA + manifest + icon 192/512 sinh từ code):
     thẻ đỏ full-screen EMERGENCY (còi 2 tông WebAudio + rung lặp +
     Notification màn hình khóa) với 3 nút **GỌI 115 · TÔI ỔN · XEM CAMERA**;
     thẻ vàng WARNING (1 tiếng ngắn, tự tắt 30s); dashboard trạng thái
     thường (score/5 module/NIHSS); lịch cảnh báo; **watchdog >15s mất
     nhịp → tự báo "MẤT KẾT NỐI"** (điện thoại phát hiện laptop chết —
     an toàn 2 chiều); WS tự reconnect 2s.
  5. **Nút TÔI ỔN → POST `/mobile/ack`** → ghi MOBILE_ACK vào Handoff
     (đối chiếu được người thân đã phản hồi lúc mấy giờ).
  6. **`/mobile/qr`**: QR chứa IP LAN thật (UDP-connect không gói tin,
     ưu tiên subnet hotspot Windows 192.168.137.x) — dashboard thêm thẻ
     "App điện thoại" + QR.
  7. `/alert/test` giờ cũng push sự kiện → test còi trên điện thoại thật.
- **Bài học khi smoke:** snapshot đầu thiếu key `type` → mobile JS không
  nhận diện được (`snap.type==='snapshot'`) — đã thêm, test 7/8→8/8.
  Ngoài ra bắt chính mình 1 lần gõ chữ Trung Quốc "mo-khoi (module)" vào HTML —
  đã sửa thành "module" (quy tắc 0 ký tự CJK của NK-17 áp cả code).
- **Kiểm định:** `src/test_mobile_alert.py` (MỚI) **8/8 PASS** (bind mặc
  định an toàn · trang mobile · manifest · icon PNG sinh code · QR · WS
  snapshot · WS nhận alert đúng seq · ack ghi handoff); regression sau
  khi đụng web_server.py: **10/10 + 58/58 PASS** (16/09).
- **Cách dùng (ghi lại để đội làm theo):** (1) laptop chạy
  `python web_server.py --lan` + bật Mobile hotspot Windows; (2) điện thoại
  kết nối hotspot, quét QR trên dashboard; (3) Chrome → Thêm vào Màn hình
  chính; (4) bấm "Bật thông báo" 1 lần để nhận cả khi tắt màn hình;
  (5) demo: dashboard bấm "Test còi + cảnh báo" → điện thoại réo + rung
  + thẻ đỏ ngay.
- **Hạn chế trung thực:** (1) điện thoại phải trong cùng mạng LAN với
  laptop (hotspot = không cần internet nhưng không báo khi ra khỏi nhà —
  là giới hạn thiết kế ghi rõ trong báo cáo); (2) Notification trên iOS
  Safari hạn chế PWA — khuyến nghị điện thoại Android cho demo; (3) còi
  WebAudio cần app đang mở hoặc thông báo đã cấp quyền.

---

## NK-26 — 17/09/2026 · KIỂM TOÁN TOÀN BỘ MODULE M1–M5 + ĐỐI CHIẾU SỐ CÔNG BỐ VỚI JSON GỐC · PHÁT HIỆN & SỬA BUG ML v3 KHÔNG CHẤM TRONG APP

**Mục tiêu:** trả lời câu hỏi "các phần trong đề tài đã ổn chưa" — kiểm tra
từng phần một, không tin lời nói, chỉ tin số chạy thật.

**1. Đối chiếu số công bố với JSON gốc (10 nguồn) — KẾT QUẢ: KHỚP 100%**
- Face v3 khóa: auc_test 0.943 / sens_test 0.8007 / spec_test 0.9409 /
  threshold 0.298 (`models/face_blend_v3_20260909_201800.json`) ✓
- Multi-seed 10 seed: 0.9398 ± 0.0099 [0.9282–0.9557] ✓
- Full-fit 100%: OOF 0.9419, threshold 0.38, note "số công bố vẫn là
  protocol khóa" ✓ — đúng nguyên tắc NK-12
- Speech LOSO người thật: LogReg 0.6202 / MLP 0.6093, sens 51.1 / spec 75.2 ✓;
  cross-mic train_head→test_array 0.4152 ✓ (p3); headMic-only 0.6688 ✓ (p2)
- openSMILE: ref 0.6202 + delta 0.0428 ✓
- Abstention: face unc_error_auc 0.777 / speech 0.488 (đúng vị trí
  selective.unc_error_auc); abstain 20% → FPR 9.5→5.3%, sens 91.1% ✓
- Early warning: FAR 0/24h cả old lẫn new; latency 10.7→5.9 /
  20.1→10.0 / 39.9→20.3 phút; ramp +18/30ph dưới ngưỡng 50: old 0/20
  vs new 20/20 @21.6ph ✓
- BlockSweep: LogReg 0.9419→0.9081 (block 50→1000) vs HistGB ≈1.0 mọi cỡ ✓
- Benchmark: cycle_total 80.9ms, camera 31.9fps, RTX 3050 6GB ✓

**2. Smoke FILE THẬT qua production path (không phải eval offline)**
- M2 Speech (đúng paths như app_family: ML 211130.pth + scaler, vosk):
  DYS F03 3 file → DANGER/DANGER/WARNING (100/100/39.8); CON FC01 3 file
  → NORMAL×3 (1.6/7.4/3.3) — PHÂN LOẠI ĐÚNG HƯỚNG trên wav thật.
- M5 Radar sim: fall/normal đều chạy, không crash.
- M1 Face: chạy 60 ảnh/lớp qua process_frame — **PHÁT HIỆN BUG (mục 3)**.

**3. BUG NK-26 — ML v3 KHÔNG BAO GIỜ chấm điểm trong app (đã sửa)**
- Triệu chứng: 120 ảnh thật qua `process_frame` → 118 được chấm nhưng
  `ml_prob` = None hết; HUD luôn hiển thị rules dù "ML v3: ON".
- Nguyên nhân: lệch TÊN KEY. `_blend_asym()` trả `asym_<tên>`
  (vd asym_browDownLeft), pose trả `yaw/pitch/roll` — nhưng artifact
  `face_blend_v3_*.json` lưu feature generic `blend_asym_00..19` +
  `pose_yaw/pose_pitch/pose_roll` (script train đặt tên generic lúc LƯU
  artifact nhưng không lưu bảng ánh xạ). `FaceMLV3.predict` tra theo tên
  artifact → thiếu → trả None → app lặng lẽ fallback rules từ SYS-29.
  Không crash, không log lỗi — chỉ im lặng (bài học: đường nối ML phải
  smoke bằng Ảnh THẬT, test đơn vị dựng dict tay nên không bắt được).
- Sửa (giữ nguyên model, giữ nguyên artifact — không vi phạm đóng băng
  model NK-12): `face_module_v7.process_frame` phát thêm alias generic
  `blend_asym_00..19` theo ĐÚNG THỨ TỰ canonical 52 blendshape (cùng quy
  tắc Left−Right cùng vòng lặp như train) + `pose_yaw/pose_pitch/pose_roll`
  khi đủ 20 cặp.
- Kiểm chứng sau sửa: ML chấm 118/118 ảnh được chấm điểm; hướng đúng —
  Stroke ML median 77.8 (mean 61.8) vs NonStroke median 6.8 (mean 11.1).
- Hồi quy sau sửa: 10/10 + 58/58 + 8/8 mobile PASS.

**4. Phát hiện phụ (chưa sửa, ghi nhận)**
- `models/vosk-model-vn-0.4` KHÔNG có trong Golden_Watch_KHKT/models
  (chỉ có ở fga_project/models) → app_family khởi tạo speech thiếu vosk:
  wpm luôn 0, không transcript; ML vẫn chấm bình thường. Cần copy/symlink
  thư mục vosk (≈40MB) trước demo speech, hoặc ghi rõ limitation.
- Ảnh Kaggle face là khung hình video cropped nhỏ (265×190) → 1/3 dính
  guard PARTIAL_FACE/NO_FACE là hành vi ĐÚNG (không chấm khi mặt lệch
  khung); trên webcam thật 1280×720 tỉ lệ này thấp hơn nhiều.
- `SpeechAnalysisModule()` khởi tạo KHÔNG paths (như 1 số test cũ) →
  rơi vào rule fallback → mọi tiếng nói đều 60.0/DANGER. Các test đã dùng
  chủ ý kiểu này ("không load model nào — nhanh") nên không phải bug app,
  nhưng smoke kiểm toán phải khởi tạo ĐÚNG như production.

**Bài học kiểm toán:** (1) mọi con số công bố phải tra được về JSON — lần
này 10/10 nguồn khớp; (2) "đã nối vào app" phải chứng minh bằng file thật
qua đúng đường khởi tạo production, không chỉ test đơn vị; (3) key của
artifact và key runtime phải có test đối chiếu 2 chiều.


## NK-27 — 17/09/2026 · VÁ HẾT VOSK THIẾU — CHUỖI SPEECH HOÀN CHỈNH (transcript + WPM chạy thật)

**Việc:** copy `vosk-model-vn-0.4` (168MB, cấu trúc chuẩn am/conf/graph/ivector)
từ `fga_project/models/` → `models/` của dự án — đúng đường app_family đã
trỏ từ trước (`os.path.join(MODELS, 'vosk-model-vn-0.4')`).

**Smoke chuỗi đầy đủ** (khởi tạo ĐÚNG như app_family: vosk + ML
`speech_torgo_20260828_211130.pth` + scaler; wav TORGO thật 16kHz):
- vosk=ON · ML=ON · scaler=ON
- DYS `F03S01_0001`: DANGER prob=100.0 · **wpm=111.6 · 24 từ · có transcript**
- CON `FC01S01_0001`: NORMAL prob=1.6 · **wpm=135.5 · 21 từ · có transcript**
- So NK-26 (chưa có vosk): cùng prob ML 100.0/1.6 — ML không đổi; thêm mới
  là transcript + wpm (trước đây luôn 0).
- Lưu ý trung thực: audio TORGO là TIẾNG ANH, model vosk là TIẾNG VIỆT →
  transcript là "tiếng Việt giả" theo âm; điều này CHỈ chứng minh cơ chế
  (nhận dạng chạy, đếm từ đúng, wpm hợp lý). Chất lượng transcript thật
  chờ dữ liệu tiếng Việt tự thu của đội (E2 — NK-20).

**Phát hiện phụ — sklearn lệch phiên bản:** nạp scaler
`speech_torgo_20260828_211130_scaler.pkl` cảnh báo
InconsistentVersionWarning (đóng gói bằng sklearn 1.9.0, máy đang 1.8.0).
Scaler vẫn nạp + kết quả ML giống hệt chạy trước đó → chưa ảnh hưởng.
VIỆC: nếu cài lại sklearn phải pin; demo máy khác phải dùng đúng env.

**Hồi quy sau thay đổi models/: 10/10 + 58/58 PASS.**

**Trạng thái kênh speech production sau NK-27:** vosk ✓ · ML ✓ · scaler ✓ ·
AGC ✓ · VAD/NO_SPEECH ✓ · baseline cá nhân ✓ — KHÔNG còn thành phần nào
thiếu file. Còn lại chỉ là dữ liệu (E2) để đo chất lượng tiếng Việt.

## NK-28 — 17/09/2026 · HUD MẶT MẠNG LANDMARK ĐỦ 478 ĐIỂM NHƯ MẪU GOOGLE + KHÓA MẶT TỨC THÌ · XƯƠNG YOLO TÔ RIÊNG CÁNH TAY + CẢNH BÁO LỆCH NHẸ

**Yêu cầu người dùng:** (1) module 1 có các điểm landmark bao quanh mặt như
file mẫu `[mediapipe_python_tasks]_face_landmarker.py`; camera góc rộng
treo góc phòng phải TỰ bắt được mặt ngay khi nhìn thấy; (2) gait/arm dùng
điểm khung xương YOLO trích xuất rõ ràng + chỉ cần lệch nhẹ là có cảnh báo.

**A. Mặt — mạng landmark chuẩn mẫu Google (web_server + app_family)**
- Dùng đúng `FaceLandmarksConnections` của mediapipe tasks API:
  TESSELATION 2556 đoạn (lưới mảnh xám 1px) + CONTOURS 124 đoạn (viền
  đậm 2px theo màu trạng thái) + LEFT/RIGHT IRIS 8 đoạn (vàng) — thay
  cho 239 chấm rời trước đây. `draw_face_mesh()` dùng chung cả 2 app.
- Lưu ý kỹ thuật: phần tử Connection phải lấy `.start/.end` (không phải
  tuple); bỏ khử răng cưa lưới → 16.5 → **11.7 ms/khung** (đo 30 lần).

**B. Mặt — khóa mặt TỨC THÌ (chỉ web_server, app giữ minh chứng)**
- Thêm `FACE_LIVE` = FaceAsymmetryDetector RIÊNG chạy trong thread camera,
  mỗi khung chẵn (~15fps): mặt xuất hiện ĐÂU trong khung là HUD bám ngay
  (<100ms), không chờ chu kỳ phân tích 2s nữa.
- Vì sao an toàn: FaceLandmarker VIDEO mode không thread-safe → instance
  riêng cho thread camera, KHÔNG đụng instance FACE của chu kỳ chấm điểm;
  ML v3 thuần toán đọc artifact nên chia sẻ được. Đường chấm điểm/alerts
  GIỮ NGUYÊN 100% (2s, HOLD 3s, median filter).
- HUD ưu tiên kết quả live có mặt; mặt vắng thì quay lại kết quả chu kỳ
  2s (giữ HOLD 3s — không nhấp nháy).

**C. Tay/chân — xương YOLO rõ + cảnh báo lệch nhẹ**
- `draw_live_pose`: đoạn CÁNH TAY (vai→khuỷu→cổ tay cả 2 bên) tô CYAN
  đậm, còn lại xanh lá; khớp tay vẽ to hơn (bán kính 5 so với 4).
- `draw_arm_panel()` mới: hiển thị góc L/R, lệch góc, lệch cao cổ tay.
- `arm_module._calculate_arm_metrics` thêm cờ `mild_asym` (lệch góc >10
  độ HOẶC chênh cao cổ tay >30px) → HUD hiện dòng vàng
  "! TAY LECH NHE — theo doi them". **Cờ CHỈ HIỂN THỊ — không đổi
  status/điểm fusion (nguyên tắc model đóng băng NK-12).** Ngưỡng 10 độ
  trùng bậc cộng điểm +10 trong rule arm (angle_asym >10) đã có sẵn.

**Kiểm chứng:** smoke vẽ trên ảnh mặt thật (mesh 478 điểm, 11.7ms) +
mild_asym True với lệch góc 20.9 độ + bảng tay vẽ được · Hồi quy
**10/10 + 58/58 + 8/8 mobile PASS**. Ảnh mẫu: `exports/nk28_mesh_demo.png`.

**Giới hạn trung thực:** (1) HUD live là HIỂN THỊ — quyết định báo động
vẫn theo chu kỳ phân tích 2s (chủ ý, không đổi); (2) camera worker giờ
chạy YOLO mỗi khung + face mỗi khung chẵn → fps ước tính ~22-25 (còn đủ
mượt, đo thật khi chạy server); (3) cờ lệch nhẹ arm chưa có số eval độc
lập — chỉ là cảnh báo sớm hiển thị.

## NK-29 — 17/09/2026 · TÀI LIỆU DẪN CHỨNG KHOA HỌC: TẠI SAO SƠ ĐỒ & SỐ LIỆU CHỈ RA ĐỘT QUỶ

**Yêu cầu người dùng:** đưa ra dẫn chứng, luận điểm, bài báo chứng minh
tại sao sơ đồ/kiến trúc đó và số liệu đó là đột quỵ (dùng cho báo cáo +
phỏng vấn).

**Sản phẩm:** `DU_CU_KHOA_HOC.md` (root) gồm 5 phần:
1. Luận điểm gốc: sơ đồ hệ = bản "không chạm" của NIHSS (Brott 1989)
   + FAST — bảng ánh xạ 1-1 module ↔ Ý 4/5/6/10 NIHSS.
2. Dẫn chứng từng module: cơ chế bệnh học (đường vỏ não/tháp/tiểu não)
   + y văn + số hệ đã đo (trích từ JSON đã đối chiếu NK-26).
3. Vì sao số đáng tin: protocol khóa, tự phát hiện leakage 2 lần
   (BlockSweep), McNemar/DeLong/ECE (NK-24), kế hoạch external
   validation — theo TRIPOD+AI (BMJ 2024) và STARD 2015.
4. Ba luận điểm 30 giây để phỏng vấn.
5. Việc phải làm sau 13/10/2026: chốt 5-6 trích dẫn nhóm [B].

**Quy ước trung thực về trích dẫn (đúng nguyên tắc không bịa):**
- [A] = chắc chắn, trích được ngay: Brott 1989 (Stroke), Lyden 2007,
  Goldberger 2000 (Circulation), DeLong 1988 (Biometrics), Eyben 2016
  (IEEE TAC, eGeMAPS), Lowe 2016 (Interspeech, TORGO), Duffy (sách
  Motor Speech Disorders), Lugaresi 2019 (MediaPipe arXiv), Johnston
  2000 (JAMA, TIA), Rothwell 2005 (Lancet, ABCD), Collins 2024
  (TRIPOD+AI), Bossuyt 2015 (STARD).
- [B] = đúng hướng NHƯNG phải xác minh DOI sau 13/10 (hạn mức tra web
  của trợ lý AI mở lại): facial palsy grading 2021-2025, FAST
  validation (Harbison), định lượng drift bằng vision, review radar
  mmWave, chi tiết FAST.AI JAMA Neurology (cần khai báo kế thừa PL1).
- Trung thực khai báo giới hạn dữ liệu: TORGO (dysarthria đa nguyên
  nhân) và PhysioNet (young/elderly/Parkinson) là dữ liệu ĐẠI DIỆN,
  không thuần đột quỵ → kiểm chứng đột quỵ thật nằm ở dữ liệu tự thu
  + video NIHSS (NK-18→23).

## NK-30 — 17/09/2026 · KIT THU DỮ LIỆU + KẾ HOẠCH 3 NGÀY (deadline cấp trường 03/10)

**Bối cảnh:** user báo deadline nộp cấp trường 03/10/2026 (trước cửa sổ
nộp trực tuyến 05–10/10). Cần hoàn thiện hồ sơ 5 món trong 3 ngày lõi.

**Sản phẩm tạo:**
1. `MAU_DONG_Y_SU_DUNG_DU_LIEU.md` — phiếu đồng ý 1 trang: 5 hoạt động
   (face/arm/gait/speech/radar), cam kết lưu 1 laptop duy nhất, không
   công bố mặt, quyền rút lui + xóa dữ liệu, ô ký của người tham gia
   và giám hộ (nếu <18).
2. `data/thu_tu_lieu/CSV_DANH_MUC_DU_LIEU.csv` — danh mục
   (loai/ten_file/ma_nguoi/tinh_huong/thoi_luong_s/ngay_thu/ghi_chu)
   + `CSV_NIHSS_2RATER.csv` — chấm mù 2 rater Ý 4/5/6/10 (r1/r2/agreed).
3. `training/ingest_video.py` — CLI trích N khung jpg chia đều thời gian
   (cv2.VideoCapture) + wav mono 16kHz (ffmpeg imageio_ffmpeg, flag
   `-vn -ac 1 -ar 16000`); nhận file hoặc thư mục; in bảng gợi ý dán CSV.
   **Smoke PASS:** video giả 5s không tiếng → 8 khung OK + cảnh báo
   không-wav đúng; video có tiếng 3s (sine 440) → 5 khung + wav
   sr=16000 mono dur=3.0s. Video giả ĐÃ XÓA.
4. `KE_HOACH_3_NGAY.md` — Bước 0 (git commit an toàn, in đơn, kiểm tra
   thiết bị) + Ngày 1 18/09 thu dữ liệu (10–15 người; face 3 tình huống,
   arm/gait MP4, speech 2 câu TV, radar 10 kịch bản) + Ngày 2 19/09 chấm
   mù 2 rater + eval NK-18→21 + khởi động 72h (hoặc 24h rút gọn — ghi
   trung thực) + đóng gói chụp BOM + backup 3 nơi + Ngày 3 20/09 làm 5
   món hồ sơ (báo cáo 15 trang từ DU_CU_KHOA_HOC + video 3 phút + PL1 +
   poster PL3 + sổ giấy) + buffer 21/09→02/10 (Protocol B rút gọn 20–30
   tình huống, nộp sớm 30/09–01/10).

**Nguyên tắc ghi trong kế hoạch:** KHÔNG retrain/đổi ngưỡng (model đóng
băng NK-12); không đòi 100% chính xác — câu chuẩn "AUC 0.94 ± 0.01,
thiếu sót đã đo và khai báo"; mọi số truy vết JSON; mọi thay đổi ghi NK-xx.

**Việc chờ:** Task #10 (smoke MP4 qua toolkit NK-05) và #11 (nạp video
YouTube) chờ user nộp dữ liệu; git commit chờ user OK.

## NK-31 — 19/09/2026 · CHỐT TÊN ĐỀ TÀI THỐNG NHẤT + COMMIT BẢN AN TOÀN + DỜI NGÀY THU DỮ LIỆU

**1. Git commit `d0749aa` "Buoi 7 (16–17/09)" — bản an toàn:** 106 file,
+21,981 dòng — toàn bộ NK-24→30 (stat tests NK-24, PWA mobile NK-25, fix
L-35 NK-26, vosk NK-27, HUD mesh + arm panel NK-28, DU_CU + kit 3 ngày
NK-29/30, trụ A/B, 10 script training/eval). `models/vosk-model-vn-0.4/`
(168MB, model bên thứ ba — copy lại từ fga_project được) đưa vào
`.gitignore`; mọi thứ khác vào repo. Working tree sạch sau commit.

**2. Thu dữ liệu 18/09 BỊ HOÃN** — `data/thu_tu_lieu/` hiện chỉ có 2 CSV
mẫu. Cần chọn ngày thu mới; các eval NK-18→21 + Protocol B rút gọn dời
theo. Đơn đồng ý + quy trình thu giữ nguyên từ KE_HOACH_3_NGAY.md.

**3. Chốt tên đề tài THỐNG NHẤT (đội chọn phương án ngắn):**
> **Golden Watch — Hệ giám sát đa cảm biến không đeo phát hiện sớm đột
> quỵ tại nhà và kiểm soát báo động giả**

Lý do đổi khỏi tên sổ cũ ("...qua tín hiệu vận động và cơ chế kiểm soát
báo động giả 4 lớp"): (a) "tín hiệu vận động" không bao hết speech +
radar; (b) thiếu "không đeo" — khác biệt cốt lõi so wearable; (c) thiếu
"tại nhà"; (d) "giám sát sức khỏe" quá rộng. "4 lớp" để báo cáo nói,
không đưa vào tên. Đã thay ở 5 file: `NHAT_KY_DU_AN.md` (trang bìa),
`BAO_CAO_DU_AN.md`, `TONG_KET_NGHIEN_CUU.md`, `PL1_khai_bao_su_dung_AI.md`,
`README.md`. GIỮ NGUYÊN 2 file lưu vết `NHAT_KY_DU_AN_v1_recovered.md` +
`TONG_QUAN_DU_AN_FINAL.md` (bản lưu lịch sử). Nếu phỏng vấn hỏi "Watch
mà không đeo?": Watch = canh giác/giám sát, không phải đồng hồ đeo.

## NK-32 — 19/09/2026 · WEB ĐỦ 5/5 MODULE (SPEECH ONLINE) + 720P + SỬA 2 BUG CHU KỲ BÁO ĐỘNG + L-36

**1. Speech vào web (`web_server.py`, thread 3 mới — kiến trúc 3→4 luồng):**
lazy init NGAY TRONG thread (dashboard mở được tức thì, vosk 168MB + torch
nạp nền sau); đúng 3 path production như app_family (`vosk-model-vn-0.4` +
`speech_torgo_20260828_211130.pth` + scaler — KHÔNG dùng bản full NK-12) +
`load_baseline(data/baselines/user_default.json)`; chu kỳ ghi 5s → phân tích
1 cửa sổ 5s → ngủ 5s (~12s/nhịp, bù nhịp dày hơn median-3-cửa-sổ của
app_family); lỗi init/mic → note + thử lại sau 30s (cắm mic sau tự lành);
NO_SPEECH KHÔNG xóa kết quả cũ (nói khó là triệu chứng dai dẳng) — giữ đến
hết hạn 90s, chỉ ghi "im lang — dang nghe". Smoke thật: vosk load OK, AGC
chạy, có transcript Vosk, speech THẬT SỰ vào fusion (log alert thấy
R1: 2/3 FAST "(arm, speech)").

**2. Camera 720p + chống trễ + bảo vệ thread:** `cap.set` 1280×720 +
`CAP_PROP_BUFFERSIZE=1` (luôn khung mới nhất) — mặt cách 3–4m tăng ~40–60px
lên ~80–120px để MediaPipe khóa được; YOLO imgsz GIỮ 480 (letterbox — chi
phí không đổi theo độ phân giải capture); toàn bộ khối vẽ bọc try/except +
`_rate_warn()` (in tối đa 1 lần/30s) — HUD hỏng không giết thread camera;
`draw_live_pose` trước đây nuốt exception IM LẶNG (không phân biệt được
"không phát hiện" với "văng lỗi") → giờ in lỗi rate-limit 30s để chẩn đoán.

**3. BUG A (nghiêm trọng, đúng kế hoạch):** `push_mobile_event('alert',
nihss_total=nih.get('total'))` dùng biến `nih` TRƯỚC khi gán (NIHSS tính ở
dưới) → `UnboundLocalError` ở MỌI chu kỳ có cảnh báo, bị try/except của
`analysis_worker` nuốt sạch → HANDOFF + đẩy mobile + `PROFILE.record` +
cập nhật STATE + heartbeat mobile BỎ — dashboard đóng băng ĐÚNG LÚC BÁO
ĐỘNG. Fix: chuyển `nih = estimate_nihss(filtered)` + CI bootstrap lên
TRƯỚC khối ALERT (giờ alert mang NIHSS thật ngay lần đầu).

**4. BUG B:** khối NIHSS + `HANDOFF.add_event('CYCLE')` bị LẶP 2 lần →
CYCLE ghi trùng + bootstrap NIHSS tính 2 lần mỗi chu kỳ. Fix: giữ ĐÚNG
1 CYCLE/chu kỳ, xóa luôn push `alert_nihss` (nhánh JS mobile giữ nguyên —
vô hại, không còn được gửi tới).

**5. L-36 (phát hiện NGAY LÚC smoke, ngoài kế hoạch —
`src/defense/personal_profile.py`):** `record()` đọc `st['days_ok']` nhưng
`status()` trả về khóa `'done'` → ngày 1–2 chỉ MAY NHỜ short-circuit của
`and` (`st['day'] >= 3` là False); 19/09 đủ ngày thứ 3 → `KeyError:
'days_ok'` ở MỌI chu kỳ (801 lần trong log smoke) → web đóng băng hoàn toàn,
modules rỗng. Fix 1 từ: `st['done']`. Sau fix: profile **day 3, done=true,
applied=true** — chế độ học 3 ngày tự hoàn thành và calibrate Defense L4
LẦN ĐẦU TIÊN chạy trọn vẹn trong production.

**6. Nối speech vào chu kỳ phân tích:** `set_audio_flag` THẬT (trước đây
False cứng "speech test ở app Streamlit"): speech tươi ≤90s VÀ prob ≥30 —
đúng ngữ nghĩa fall-AND-audio như app_family; `mods['speech']` chỉ đưa khi
còn tươi — hết hạn fusion tự cân lại trọng số (face .20/speech .20/arm
.30/gait .15/radar .15 — fusion/defense/NIHSS không sửa gì); dashboard thêm
thẻ "🗣️ Giọng nói — liên tục" (status, điểm /100, từ/phút, số từ, tuổi kết
quả, transcript Vosk escape HTML), JS name map thêm `speech`; mobile PWA
`nm` thêm `speech:'Giong noi'` (snapshot WS tự mang nguyên khối modules).

**7. Smoke thật sau fix:** 5/5 module hiện đủ; nhịp chu kỳ 2.2s chuẩn
(20s → +9 chu kỳ); 0 lỗi [ANALYSIS]; radar SIM đổi scenario OK; scenario
ngã/alert → `last_alert` WARNING 55.0 chạy qua nhánh BUG A KHÔNG crash,
NIHSS item10_dysarthria=2 từ speech, `analysis_count` tiếp tục tăng.

**8. Hồi quy (server TẮT trước test mobile):** `test_sys14b_regression.py`
**10/10** + `test_all_metrics.py` **58/58** + `test_mobile_alert.py` **8/8**
PASS. Model/ngưỡng/conf mặt KHÔNG đổi — model đóng băng NK-12 giữ nguyên.

**9. Quan sát demo (chưa sửa — chờ người nói thật):** mic laptop để xa →
AGC khếch đại cả tiếng ồn nền (peak 0.000 → x4915) cho speech DANGER 98.4
GIẢ + Vosk "bịa" từ trên nhiễu — demo phải để người nói gần mic; gait
WARNING 50 khi đối tượng NGỒI (pose chập người) — chuẩn bị lời giải thích
khi phỏng vấn. Việc commit cuối ngày gồm cả đổi tên đề tài + NK-31/32
(theo dặn để cuối ngày 19/09).
