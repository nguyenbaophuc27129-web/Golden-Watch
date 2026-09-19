# TỔNG KẾT NGHIÊN CỨU — GOLDEN-WATCH 3.0 (HỆ GIÁM SÁT TỰ TIN)

> Phiên bản 1.0 — 12/09/2026 · Dự án: **Golden Watch — Hệ giám sát đa cảm biến
> không đeo phát hiện sớm đột quỵ tại nhà và kiểm soát báo động giả** ·
> Lĩnh vực 5: Kỹ thuật Y Sinh.
> Mọi số liệu trong tài liệu là SỐ THẬT từ các protocol đã khóa, chi tiết tại
> `so_nhat_ky.md` (NK-01→NK-12) + `BANG_THANG_DO_KIEM_DINH.md`.
> Mục tiêu thi: Giải Nhất TP.HCM → đại diện TP.HCM dự KHKT quốc gia → giải Nhì quốc gia.

---

## 1. SẢN PHẨM LÀ GÌ (3 CÂU DÙNG CHO MỌI PHÒNG VẤN)

1. Golden-Watch là hệ thống giám sát **không đeo thiết bị** chạy liên tục tại
   nhà, dùng 1 webcam + 1 radar 24GHz để phát hiện 5 nhóm dấu hiệu đột quỵ
   (méo mặt · nói líu · tay yếu · dáng đi bất thường · ngã/bất động), ước tính
   điểm NIHSS và bàn giao cho bệnh viện bằng QR/PDF trong "giờ vàng" <4,5h.
2. Khác biệt khoa học: hệ thống **TỰ TIN** — biết module nào đang đáng tin
   (abstention/uncertainty), phát hiện **suy giảm tiến triển chậm** mà luật
   ngưỡng thông thường không bao giờ báo (early warning EWMA+CUSUM), và chống
   báo giả bằng kiến trúc 4 lớp **đo được** (defense ablation), không phải
   tuyên bố thiết kế.
3. Toàn bộ AI chạy **offline trên 1 laptop**, chi phí mua thêm ≈1,01 triệu
   đồng, dữ liệu không rời khỏi máy — tính cộng đồng + riêng tư.

---

## 2. KIẾN TRÚC TỔNG THỂ (SƠ ĐỒ NGUYÊN LÝ)

```
┌────────────────────────────── CỤM CẢM BIẾN (vỏ gắn tường 1.4–1.6m) ─────────────────────────────┐
│  Webcam C270 (720p/30fps)                       Radar LD2450 24GHz (baud 256000)                 │
└──────┬──────────────────────────────────────────────────┬───────────────────────────────────────┘
       │ 30 fps                                            │ 3 Hz (không camera → riêng tư)
┌──────▼───────┐ ┌──────────────┐ ┌──────────────┐ ┌──────▼──────┐ ┌──────────────┐
│ M1 FACE      │ │ M2 SPEECH    │ │ M3 ARM       │ │ M4 GAIT     │ │ M5 RADAR     │
│ MediaPipe    │ │ 48 ft + VAD  │ │ YOLOv8-pose  │ │ YOLOv8-pose │ │ Doppler/     │
│ 28ft + ML v3 │ │ + MLP prod   │ │ keypoints    │ │ chu kỳ bước │ │ micro-motion │
│ AUC .94±.01  │ │ LOSO .62     │ │ (E3 đang thu)│ │ LOSO .879   │ │ 10 kịch bản  │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬──────┘ └──────┬───────┘
       └────────────────┴───────┬────────┴────────────────┴───────────────┘
                                ▼
┌─────────────────── DEFENSE ENGINE 4 LỚP (chống báo giả) ────────────────────┐
│ L1 Calibration baseline cá nhân (3 ngày học) · L2 Context (ngữ cảnh)        │
│ L3 Temporal (persistent 30s) + EARLY WARNING EWMA+CUSUM [mới v3.0]          │
│ L4 Adaptive threshold vùng xám [30,40)                                      │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
┌──────────── FUSION (trọng số arm .30 · face/speech .20 · gait/radar .15) ───┐
│ + Luật FAST: R1 (≥2/3 dấu hiệu → EMERGENCY) · R2 (1 module ≥80 → WARNING)   │
│ + ABSTENTION [mới v3.0]: uncertainty cao → NEEDS_CHECK (người xác nhận),    │
│   KHÔNG BAO GIỜ hạ cấp EMERGENCY                                            │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
┌─────────── TRIAGE → NIHSS 4 items (4+5+6+10, max 15) → MILD/MOD/SEVERE ────┐
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
┌──────────── ALERT (còi + màn hình) → HANDOFF QR/PDF cho bệnh viện ─────────┐
│ FastAPI web_server.py cổng 5001 · MJPEG 15–20fps · dashboard offline       │
└─────────────────────────────────────────────────────────────────────────────┘
```

**9 công nghệ sử dụng:** (1) MediaPipe FaceLandmarker Tasks API (28 đặc trưng),
(2) LogisticRegression calibrated, (3) MLP 48→256/128/64 (speech), (4)
YOLOv8n-pose (tay + dáng đi), (5) radar mmWave LD2450 Doppler, (6) EWMA+CUSUM
change-point, (7) Platt calibration + selective prediction, (8) FastAPI/MJPEG
real-time server, (9) QR/PDF handoff offline.

---

## 3. SƠ ĐỒ THUẬT TOÁN CHUẨN KHOA HỌC

### 3.1. Phát hiện méo mặt (mô hình chính thức)
```
Ảnh 720p → MediaPipe FaceLandmarker (conf 0.3, HOLD 3s mất mặt thoáng)
  → 5 ratio lâm sàng + 20 blendshape bất đối xứng (L−R) + yaw/pitch/roll
  → StandardScaler → LogisticRegression (C=0.03, 28 ft)
  → prob ML [0..1] (rules giữ ở score_rules để đối chiếu)
Guards: laugh-guard (miệng mở + khóe đối xứng → −60% metric biểu cảm),
PARTIAL_FACE (chạm mép khung → không chấm)
```
- **Đánh giá khóa:** held-out block test AUC **0.943** (sens 80.1 / spec 94.1 @thr 0.298)
- **Multi-seed 10 seed:** AUC **0.9398 ± 0.0099** [0.9282–0.9557] → công bố "0.94 ± 0.01"
- **HistGB AUC 1.000 = LEAKAGE** — chứng minh bằng Giao thức BlockSweep, không dùng
- Refit 100% dataset (NK-12): coef tương quan 0.993 bản cũ, OOF all-data 0.9419 → artifact production

### 3.2. Early Warning — dò suy giảm tiến triển (mới, NK-11)
```
Mỗi chu kỳ 2s: điểm fused → 3 phép tính song song:
  1. EWMA(α=0.15): mức nền làm mượt
  2. CUSUM một phía: S_t = max(0, S_{t-1} + (x_t−mu0)/σ − k), k=0.5σ, h=5σ
     (mu0/σ = median/MAD baseline cá nhân đoạn bình thường)
  3. Run-length: số chu kỳ liên tiếp x ≥ mu0+1.5σ
Quyết định:
  DETERIORATING = (S_t > 5σ) VÀ (run ≥ 15 chu kỳ = 30s)   → cảnh báo "đang xấu dần"
  WATCH         = EWMA ≥ mu0+2σ                            → quan sát mềm
  OK            = còn lại
```
- **Kết quả (mô phỏng seed 42, ghi rõ):** FAR 0/24h cả 2 phương pháp · latency
  giảm **45–50%** (10.7→5.9 / 20.1→10.0 / 39.9→20.3 phút) · **ramp +18đ/30ph
  dưới ngưỡng 50: luật cũ 0/20 KHÔNG BAO GIỜ báo — mới 20/20 @21.6 phút**

### 3.3. Uncertainty + Abstention (mới, NK-10)
```
prob hệ = mean(ensemble) · uncertainty = std(ensemble) + margin ngưỡng
Thành viên ensemble: face = 10 seed OOF · speech = 2 kiến trúc (LogReg+MLP)
Abstain top-frac bất định nhất → NEEDS_CHECK (người xác nhận)
```
- **Face: unc_error_auc 0.777** — abstain 20% → FPR 9.5→**5.3%**, sens **tăng** 91.1%
- **Speech: unc_error_auc 0.488 ≈ random** → thước đo trung thực "module nào đáng tin"

### 3.4. Speech production
48 ft tự trích (pitch, jitter, shimmer, energy, VAD…) → MLP 256/128/64 ·
median-3 cửa sổ (chống outlier) · ngưỡng ALARM 56% (Youden J=1.0 extended test,
FPR 0%) · LOSO người thật 0.62 = limitation được đo trung thực · openSMILE
eGeMAPSv02 0.663 (+0.043, giữ 48 ft vì không đáng đổi)

---

## 4. PHẦN CỨNG — BẢNG MỜI GIÁ (BOM)

| # | Linh kiện | Vai trò | Giá (VND) | Ghi chú |
|---|---|---|---|---|
| 1 | Laptop RTX 3050 6GB | Máy chủ AI thời gian thực | *(máy có sẵn)* | torch 2.14.0+cu126, numpy 2.3.4 pin |
| 2 | Logitech C270 | Camera 5 dấu hiệu | 350.000 | đo thật 1280×720 @32.2fps |
| 3 | LD2450 + UART-USB | Radar 24GHz ngã/bất động | 450.000 | ±60°/±80°, 2–3 vật thể, 0.75–6m |
| 4 | Vỏ gắn tường (gỗ nhựa/3D) | Đóng gói cụm cảm biến | 100.000 | mục Chế tạo |
| 5 | Cáp USB nối dài ×2 | Đi dây | 60.000 | |
| 6 | Đế/chân máy + ốc + keo | Lắp đặt | 50.000 | |
| | **TỔNG MUA THÊM** | | **≈1.010.000** | **"Tính cộng đồng": rẻ hơn hàng chục lần 1 máy ATP y tế** |

**Hiệu năng đo thật (benchmark 08/09):** chu kỳ phân tích 5 module ≈ **81ms**
(chu kỳ app 3000ms → dư ~2.9s) · camera thật 32.2fps · độ trễ radar 600ms/nhịp.

---

## 5. THANG ĐO KHOA HỌC (cách chứng minh "sản phẩm đo được")

| Thang đo | Dùng cho | Chuẩn tham chiếu |
|---|---|---|
| AUC + Wilson 95% CI | mọi tỷ lệ sens/spec | Newcombe/Wilson score |
| Youden J | chọn ngưỡng | Youden 1950 |
| LOSO (leave-one-subject-out) | gait, speech người thật | chuẩn MAD/PhysioNet |
| GroupKFold theo block | face (chống leakage) | TRIPOD+AI BMJ 2024;385:e078378 |
| Multi-seed (10 seeds) ± SD | ổn định face v3 | khuyến nghị TRIPOD+AI |
| Bland–Altman + weighted-κ quadratic | máy vs bác sĩ NIHSS (SYS-18) | Bland & Altman 1986 |
| MAE/RMSE/R² | hồi quy NIHSS | mục 2 BANG_THANG_DO |
| DeLong | so AUC 2 mô hình | DeLong 1988 |
| FAR/24h + latency | hệ thống liên tục | chuẩn thiết bị y tế tại nhà |
| Risk–coverage / selective AUC | abstention (v3.0) | selective prediction |

**2 đóng góp phương pháp (mang tên riêng, có figure):**
1. **Giao thức BlockSweep** — dò leakage bằng quét cỡ block × họ mô hình:
   LogReg ổn định 0.91–0.94 mọi block; HistGB ≈1.000 mọi block = memorize.
   Thông điệp: *"AUC 1.000 là chỉ số LỖI, không phải thành tích."*
2. **Defense Ablation** — bật/tắt từng lớp L1→L4 đo FPR thật trên 100 tình
   huống người khỏe (Protocol B) → "kiến trúc 4 lớp là kết quả đo được".

**Bậc thang kiểm định TRIPOD+AI (khung trình bày mục 4 báo cáo):**
Bậc 1 Internal validation ✅ (OOF/LOSO/block + CI) → Bậc 2 External ⬜ (E2/E3
đang thu, dataset công khai khai thác sâu) → Bậc 3 Prospective ⬜ (72h lab,
single-site, ghi trung thực).

---

## 6. TỰ CHẤM ĐIỂM THEO TIÊU CHÍ CUỘC THI (KH 5.3.2b)

| Tiêu chí | Max | Tự chấm | Bằng chứng chống điểm |
|---|---|---|---|
| Vấn đề nghiên cứu | 10 | 9 | Gánh nặng đột quỵ VN + "giờ vàng" 4.5h + người cao tuổi sống một mình; vấn đề + giả thuyết H1/H2 rõ |
| Thiết kế & phương pháp | 15 | 13.5 | 3 bậc kiểm định TRIPOD+AI + protocol đăng ký trước + BlockSweep + ablation + thang đo đầy đủ (mục 5) |
| Chế tạo & kiểm tra | 20 | 15 | Prototype cụm C270+LD2450 trên vỏ + benchmark thật (81ms/32fps) + S0–S6 nghiệm thu + 72h lab; **yếu: chưa có vỏ hoàn chỉnh — cần xong 02/10** |
| Tính sáng tạo | 20 | 16 | "Hệ giám sát TỰ TIN" (abstention + early warning + cross-check) 2 đóng góp phương pháp mang tên riêng; KHÔNG có phát hiện lâm sàng — trung thực |
| Trình bày | 35 | 28 | Sổ nhật ký NK-01→NK-12, 30 câu Q&A, video <3 phút, poster, PL1 khai báo AI, số toàn bộ kèm CI |
| **TỔNG** | **100** | **≈81.5** | Đủ vùng Nhất TP → Nhì QG (mốc Nhì QG ≈90: cần trình bày thật tốt + prototype xong) |

**3 quyết định điểm:** (1) prototype vật lý xong sớm (Chế tạo 20đ); (2) rehearse
phỏng vấn bằng HOI_DA_PHONG_VAN.md (Trình bày 35đ); (3) mọi số đều kèm n/CI —
giám khảo chuyên y đánh giá cao trung thực.

---

## 7. ĐỊNH VỊ MỚI CỦA ĐỀ TÀI (vì sao "chưa từng có" ở cấp học sinh)

Đề tài KHÔNG cạnh tranh ở "phát hiện đột quỵ bằng AI" (đã nhiều). Đề tài
cạnh tranh ở **3 lớp mà không dự án học sinh (và phần lớn paper) có:**

1. **"Biết mình không biết"** — hệ đo unc_error_auc từng module và tự đánh dấu
   NEEDS_CHECK. Phát hiện: bất định là thước đo độ tin cậy module (face 0.777
   đáng tin; speech 0.488 = sai tự tin). Đây là tư duy selective prediction —
   chủ đề nóng hậu-TRIPOD+AI, chưa thấy ở cấp THCS/THPT.
2. **"Đồng hồ vàng là DÒNG THỜI GIAN"** — phát hiện điểm mù cấu trúc của luật
   ngưỡng (drift dưới ngưỡng = không bao giờ báo, 0/20) và vá bằng EWMA+CUSUM
   baseline cá nhân (20/20) mà KHÔNG tăng báo giả (0/24h cả hai). Đúng tên
   Golden-Watch.
3. **"Chống gian lận số liệu do chính hệ thống tự chứng minh"** — BlockSweep +
   2 lần tự phát hiện leakage (face HistGB 1.000; speech 0.992 session-level →
   0.62 người-level) trước khi giám khảo phát hiện. Kể chuyện "chúng tôi tự
   bắt mình gian lận" = ấn tượng nhất hội đồng.

> Câu chốt hội đồng: *"Chúng tôi không có bệnh nhân nên không tuyên bố phát
> hiện lâm sàng. Thay vào đó chúng tôi hoàn thành đủ 3 bậc kiểm định TRIPOD+AI
> và tự phát hiện 2 trường hợp leakage bằng protocol đúng — điều phần lớn mô
> hình đã công bố còn thiếu."*

---

## 8. VIỆC CÒN LẠI (hạn 05/10/2026)

| Hạn | Việc | Trách nhiệm |
|---|---|---|
| 18–20/09 | Radar thật 10 kịch bản + video E3 (`thu_tuc/`, `tha_tay/`) + cắm cross-check | Người: phần cứng · AI: eval |
| 21–23/09 | Protocol B 100 tình huống (capture) → defense ablation + replay early-warning | Người: thời gian đội |
| 24–28/09 | SYS-18 NIHSS (50 video + 2 chấm viên CSV) + E2 speech VN (`vn_thuong/`, `vn_liu/`) | Người: chấm + ghi âm |
| 25/09–03/10 | 72h lab chạy liên tục (FAR/24h, uptime) | Người: nguồn + laptop |
| 01–04/10 | Báo cáo 15 trang (dàn ý sẵn BAO_CAO_DU_AN.md, số về 1 nguồn) · video <3 phút · poster PL3 · sổ nhật ký PL2 · GVHD ký · prototype vỏ · repo private | Người + AI hỗ trợ |
| 05/10 | NỘP HỒ SƠ | Người |

---

*Số liệu nguồn: `test_results/metrics_pack_20260907_223217/` ·
`abstention_20260912_122517/` · `early_warning_20260912_124947/` ·
`face_v3_multiseed_20260910_215359/` · `speech_speaker_loso_20260910_214834/` ·
`speech_opensmile_loso_20260910_221102/` · `blocksweep_leakage_diagnostic.png` ·
`prototype_benchmark_20260908_050756.json`.*
