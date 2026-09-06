# FILE LOGIC GIẢI THÍCH — TEST SUITE TOÀN HỆ THỐNG (src/test_all_metrics.py)
> Tài liệu học cho phỏng vấn KHKT 2026 — tạo 06/09/2026
> Chạy: `venv/Scripts/python.exe src/test_all_metrics.py` → **29/29 PASS** (06/09)
> Không cần camera/micro/dataset — audio tổng hợp (sine/silence/noise) + mock data

---

## 1. TẠI SAO CẦN TEST SUITE RIÊNG?

- Test thật (test_module2_full.py chạy TORGO) mất ~10 phút + cần dataset.
- Suite này kiểm tra **LOGIC THUẦN** các thang đo trong vài chục giây:
  chạy được SAU MỖI lần sửa code (regression test) mà không tốn tài nguyên.
- Nguyên tắc: **test biên (boundary)** — đặt giá trị sát ngưỡng (29.9, 30, 49.9,
  50, 69.9, 70) thay vì giá trị giữa để bắt lỗi `<` vs `<=`.

---

## 2. 5 SECTION VÀ THANG ĐO KIỂM TRA

### A. Module 2 Speech (11 test)
| Test | Thang đo | Kỹ thuật |
|---|---|---|
| A1-A3 | VAD speech_ratio | silence→0, tone→1.0, 50/50→~0.5±0.15 |
| A4 | WPM | 25 từ/10s = 150 (công thức words/duration×60) |
| A5 | NIHSS item 10 | biên 30/50/70 → 0/1/2/3 |
| A6-A8 | Window selection (M2-10) | audio 3 lớp (10s im + 15s tone + 5s noise) → window phải rơi vùng tone, ratio>0.9 |
| A9 | Features | đúng 48, không NaN, pitch tone 440Hz đo ra ~440±25 |
| A10 | Baseline | set→save JSON→load→so wpm |
| A11 | NO_SPEECH | im lặng → status NO_SPEECH + prob reset 0 (không báo động giả từ im lặng) |

### B. FusionEngine (8 test)
| Test | Logic | Kỹ thuật |
|---|---|---|
| B1 | Adapter key | 4 module đặt tên prob khác nhau (score/speech_prob/arm_prob/gait_prob) |
| B2 | Renormalize | face 40 + speech 10, còn lại 0.4 → (8+2)/0.4 = **25** (tính tay) |
| B3 | Skip invalid | NO_SPEECH prob 90 vẫn bị loại → chỉ arm 50 → fused 50 |
| B4 | Luật R1 | face 60 + arm 60 (weighted 45.7 < 50) → R1 nâng 75 EMERGENCY |
| B5 | Luật R2 | speech 85 đơn lẻ → floor 55 |
| B6 | Biên risk | 29.9→NORMAL, 30→MONITOR, 50→WARNING, 70→EMERGENCY (single module, renorm = chính nó) |
| B7 | NIHSS severity | total = sum items; 5=MILD, 6-13=MODERATE, 14+=SEVERE |
| B8 | Trend | Δ(window) > +10 WORSENING, < −10 IMPROVING, giữa = STABLE |

### C. AlertSystem (4 test)
- C1: quyết định mức 80/50 (score 95 + risk NORMAL → vẫn EMERGENCY = defense in depth)
- C2: cooldown **per-level** — WARNING vừa phát không chặn EMERGENCY (nguy hơn phải xuyên qua)
- C3: JSONL ghi ra file + parse ngược được (crash-safe append)
- C4: None/{} không crash

### D. Model checkpoints (4 test)
- D1: 7 file bắt buộc tồn tại (speech/arm/gait .pth + scaler + face_landmarker_v2.task)
- D2-D3: state_dict parse kiến trúc như loader thật (pattern `network.{idx}.weight`, idx+=4) — bắt lỗi model train kiến trúc khác với code load
- D4: scaler `n_features_in_` — speech=48 (khớp features), arm=16

### E. Artifacts (2 test)
- E1: JSON kết quả extended M2 mới nhất: window_mode=median3, TPR≥90, FPR_optimal<5
- E2: alert JSONL có event

---

## 3. SỐ LIỆU FLASH CARD

| Số liệu | Giá trị |
|---|---|
| Tổng test | **29** (A:11, B:8, C:4, D:4, E:2) |
| Kết quả 06/09 | **29/29 PASS** |
| Speech features | **48** (MFCC 39 + pitch 4 + energy 3 + ZCR 2) |
| Arm scaler features | **16** |
| Speech arch | 48→256→128→64→2 |
| Extended M2 (median3) | TPR 96.3%, FPR 0% @th56, Youden J=1.00 |
| Fusion weights | arm .30, face/speech .20, gait/radar .15 |
| Risk biên | 30 / 50 / 70 |

---

## 4. Q&A PHỎNG VẤN

**Q1: Tại sao dùng audio tổng hợp thay vì dữ liệu thật?**
A: Unit test phải **deterministic** — tone 440Hz luôn ra pitch 440, silence luôn
ratio 0. Dữ liệu thật cho kết quả dao động mỗi lần chạy (flaky test). Dữ liệu
thật đã có test riêng (test_module2_full.py trên TORGO) — hai tầng bổ sung nhau.

**Q2: Làm sao test window selection mà không có audio dài thật?**
A: Ghép 3 đoạn tổng hợp: 10s im + 15s tone + 5s noise-yếu. Nếu thuật toán đúng,
cửa sổ 5s tốt nhất BẮT BUỘC rơi vào đoạn tone (start 9–20.5s, ratio >0.9).
Đây là test hành vi (behavior test), không phụ thuộc dữ liệu.

**Q3: Test biên quan trọng nhất ở đây là gì?**
A: B6 — 6 giá trị sát ngưỡng (29.9/30, 49.9/50, 69.9/70). Nếu code viết
`>` thay vì `>=` thì 30→NORMAL (sai) thay vì MONITOR. Đây là loại bug khó
thấy nhất khi nhìn code, dễ thấy nhất khi test biên.

**Q4: Vì sao parse kiến trúc từ state_dict trong test D2-D3?**
A: Tái tạo đúng logic auto-detect của loader thật (M2-02 fix). Nếu ai train
model kiến trúc mới mà loader không đọc được, test báo ngay trước khi đến
runtime. idx+=4 vì mỗi block là Linear+BatchNorm+ReLU+Dropout.

**Q5: Suite này có thay thế được test trên TORGO không?**
A: Không. Suite kiểm tra LOGIC ĐÚNG (implementation correctness); TORGO kiểm
tra HIỆU QUẢ THỰC (TPR/FPR). Logic đúng chưa chắc model tốt — hai tầng
khác nhau của chất lượng phần mềm ML.
