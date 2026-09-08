# PSCS Golden-Watch v8.0

**Pre-hospital Stroke Care System** — Hệ thống sàng lọc tiền viện hỗ trợ phát hiện
sớm đột quỵ bằng AI, cho người cao tuổi tại nhà. Dự án nghiên cứu khoa học kỹ
thuật học sinh THPT.

> ⚠️ **KHÔNG PHẢN CHẤN ĐOÁN** — công cụ SÀNG LỌC HỖ TRỢ. Luôn hiển thị khuyến
> cáo "GỌI 115". Bác sĩ mới là người chẩn đoán và chấm NIHSS chuẩn.

## Kiến trúc

```
[M1 Face]  [M2 Speech]  [M3 Arm]  [M4 Gait]  [M5 Radar]
 MediaPipe   TORGO+ML    YOLO-pose  YOLO-pose  LD2450 24GHz
    │            │           │          │          │
    └────────────┴─────┬─────┴──────────┴──────────┘
                       ▼
            [DefenseEngine 4 lớp]   ← L1 calibrate / L2 context
                       │              L3 temporal / L4 adaptive
                       ▼
                [FusionEngine]       ← trọng số + luật FAST R1/R2
                       │
         ┌─────────────┼──────────────┐
         ▼             ▼              ▼
  [NIHSS Estimator] [Triage]   [AlertSystem v2]
   4 items /13       subtype     buzzer + JSONL
   Monte Carlo CI    severity    + APP FEED (Streamlit)
         │           hospital          │
         └───────────┴───────┬─────────┘
                             ▼
                  [Handoff QR/PDF] ──► [App gia đình Streamlit]
```

## 5 module phát hiện

| Module | Dấu hiệu | Công nghệ | Độ chính xác |
|---|---|---|---|
| M1 Face | Méo mặt, sụp mí | MediaPipe FaceLandmarker | 93.75% (n=2783) |
| M2 Speech | Nói khó, nói đơ | 48 features + MLP, TORGO | 83.07% / TPR 96.3% FPR 0% @th56 |
| M3 Arm | Tay yếu, rơi tay | YOLOv8-pose + ML | 99.50% |
| M4 Gait | Đi lệch, mất thăng bằng | YOLOv8-pose + ROC opt | 86.13% |
| M5 Radar | Ngã, bất hoạt | HLK-LD2450 24GHz UART | proxy fall detection |

## Chạy nhanh

```bash
# 1. Cài dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Tải model weights (xem models/README.md) hoặc dùng bản đã train

# 3. App gia đình — giám sát liên tục + báo động in-feed
streamlit run app_family.py

# 4. Chạy test toàn bộ thang đo + logic (58 cases, không cần camera/radar)
venv/Scripts/python.exe src/test_all_metrics.py
```

## App gia đình (Golden-Watch Family)

`app_family.py` — Streamlit dành riêng cho người thân:

- **Tab Giám sát**: camera + radar quét liên tục, defense → fusion → NIHSS
- **Tab Báo động**: tin nhắn in-feed kèm hướng dẫn thời gian vàng (<4.5h,
  ~1.9 triệu neuron/phút — Saver 2006), khuyến cáo GỌI 115
- **Tab Kết quả & NIHSS**: 4 items ước tính + khoảng tin cậy 95%
- **Tab Bệnh viện & Handoff**: đề xuất tuyến + báo cáo QR/PDF cho nhân viên y tế
- **Tab Kiểm tra nói**: median-3 consensus (M2-10)

## Cấu trúc repo

```
src/
  detection/    face_module_v7, speech_module_v2, arm_module, gait_module, radar_module
  fusion/       fusion_engine, nihss_estimator, triage_engine, validation_metrics
  defense/      defense_engine (4 lớp giảm false alarm)
  alerts/       alert_system v2 (app in-feed)
  handoff/      handoff_system (timeline + QR + PDF)
  training/     huấn luyện ML
  test_all_metrics.py   test suite 58 cases
module1/..module5/      app GUI từng module
app_family.py           Streamlit app gia đình
scripts/                generate_science_table.py (SYS-06)
docs/kiem_dinh_y_khoa/  quy trình kiểm định + mẫu thư bác sĩ
```

## Kiểm định y khoa

Trạng thái trung thực: **CHƯA có validation lâm sàng** (đang chuẩn bị, xem
`docs/kiem_dinh_y_khoa/QUY_TRINH_KIEM_DINH_Y_KHOA.md`):

- Phần A: 50 video NIHSS công khai — target r ≥ 0.85, MAE < 2
- Phần B: 100 kịch bản tại nhà — Sens > 90%, Spec > 95%, FPR < 5%
- Phần C: 2 thư xem xét bác sĩ (BV Nhân dân 115, BV ĐHYD TP.HCM)

Bảng thang đo: `BANG_CO_SO_KHOA_HOC.xlsx` — đánh dấu trung thực
VERIFIED (có DOI/PMID) / CANONICAL (kinh điển, cần verify) / SELF-DESIGN
(cần validate lâm sàng).

## Đạo đức nghiên cứu

1. Không dùng trên người bệnh thật khi chưa có chấp thuận của bác sĩ
2. Dữ liệu nghiên cứu ẩn danh
3. Disclaimer "KHÔNG PHẢN CHẨN ĐOÁN — GỌI 115" luôn hiển thị
4. Thư bác sĩ là consultation/review, KHÔNG phải clinical trial

## Tác giả

PSCS Team — dự án NCKHKT dành cho học sinh THPT, 2026.

---

## PHIÊN BẢN ĐÓNG GÓI KHKT (Golden_Watch_KHKT)

Bộ này là **code chính thức của đề tài** gom từ repo làm việc `fga_project/`
ngày 06/09/2026, đã compile-check PASS toàn bộ.

### Cấu trúc

```
Golden_Watch_KHKT/
├── app_family.py            # App gia đình Streamlit (sản phẩm chính - demo thi)
├── src/
│   ├── detection/           # face_v7, speech_v2, arm, gait, radar (LD2450)
│   ├── fusion/              # fusion_engine, nihss_estimator, triage, validation_metrics
│   ├── defense/             # DefenseEngine 4 lớp giảm false alarm
│   ├── alerts/              # AlertSystem v2 (app in-feed)
│   ├── handoff/             # Timeline + QR + PDF bàn giao y khoa
│   └── test_all_metrics.py  # Test suite 58/58 PASS (không cần camera/radar)
├── module1..4/              # App GUI từng module (module5 = radar trong src/detection)
├── training/                # Script huấn luyện (face/speech/gait) — bằng chứng phương pháp
├── scripts/                 # Sinh bảng cơ sở khoa học (SYS-06)
├── docs/kiem_dinh_y_khoa/   # Quy trình kiểm định + mẫu thư bác sĩ
├── models/                  # Xem models/README.md — copy weights để chạy
├── LOI_SO_MODULE.md         # SỔ NHẬT KÝ NGHIÊN CỨU (bắt buộc theo thể lệ — trình bày khi phỏng vấn)
└── BANG_CO_SO_KHOA_HOC.xlsx # Thang đo + nguồn (VERIFIED/CANONICAL/SELF-DESIGN)
```

### Lưu ý với giám khảo / người xem bộ code

- Sản phẩm KHÔNG PHẢN CHẤN ĐOÁN — công cụ sàng lọc hỗ trợ, luôn khuyến cáo GỌI 115
- Trạng thái trung thực: speech = dữ liệu thật (TORGO công khai); face = dữ liệu
  tự thu; arm/gait/radar = dữ liệu mô phỏng, kiểm định lâm sàng đang chuẩn bị
  (docs/kiem_dinh_y_khoa/)
- Test nhanh không cần phần cứng: `python src/test_all_metrics.py` → 58/58 PASS
