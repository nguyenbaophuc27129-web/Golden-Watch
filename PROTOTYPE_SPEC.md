# PROTOTYPE PSCS "GOLDEN-WATCH" — SẢN PHẨM MẪU NGHIÊN CỨU (v1.0 — 08/09/2026)

> Cấu hình chính thức do đội chốt: **laptop GPU NVIDIA RTX 3050 làm máy chủ xử lý
> trung tâm + webcam Logitech C270 + radar mmWave LD2450**.
> Mục tiêu hồ sơ: đóng gói thành 1 sản phẩm mẫu (prototype) minh chứng mục
> "Chế tạo và kiểm tra" (20đ — KH 5.3.1b/5.3.2b).

## 1. KIẾN TRÚC PHẦN CỨNG

```
┌──────────────────────────────────────────────────────────────┐
│                MÁY CHỦ — Laptop NVIDIA RTX 3050 (6GB)        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Phần mềm: app_family.py (Streamlit)                   │  │
│  │  - MediaPipe FaceLandmarker (méo mặt)      → CPU       │  │
│  │  - YOLOv8n-pose (tay yếu/dáng đi)          → GPU CUDA  │  │
│  │  - Vosk-vn 0.4 + MLP speech (nói khó)      → CPU       │  │
│  │  - Fusion + Defense + NIHSS + Alert        → CPU       │  │
│  └────────────────────────────────────────────────────────┘  │
│        ▲ USB                          ▲ USB-CDC @256000 baud  │
└────────┼──────────────────────────────┼───────────────────────┘
         │                              │
┌────────┴─────────┐          ┌─────────┴──────────────┐
│ Logitech C270    │          │ LD2450 24GHz mmWave    │
│ 720p, gắn tường/ │          │ (UART-USB, quét 2 tầng │
│ màn hình, hướng  │          │ fresnel, 3 vật thể,    │
│ người bệnh       │          │ micro-motion phát ngã) │
└──────────────────┘          └────────────────────────┘
```

## 2. BOM (BILL OF MATERIALS) — bảng giá đưa vào báo cáo

| # | Linh kiện | Vai trò | Giá tham khảo (VND) | Ghi chú |
|---|---|---|---|---|
| 1 | Laptop RTX 3050 6GB | Máy chủ suy luận AI thời gian thực | (tận dụng máy có sẵn) | i5/i3, RAM ≥8GB, SSD ≥256GB; **bắt buộc cài torch bản CUDA** (hiện máy đang torch CPU — xem §5) |
| 2 | Logitech C270 | Camera méo mặt/tay/dáng đi | ~350.000 | 720p@30fps, FOV 60°, hoạt động 0–35°C |
| 3 | LD2450 + board UART-USB | Radar phát ngã/bất động (không camera, bảo vệ riêng tư) | ~450.000 | 24GHz, ±60°/±80°, baud 256000, 2/3 vật thể, micro-motion |
| 4 | Vỏ/hộp gắn tường (in 3D hoặc gỗ nhựa) | Đóng gói camera + radar thành 1 cụm | ~100.000 | PR-01 |
| 5 | Cáp USB nối dài ×2 | Đi dây tới vị trí người bệnh | ~60.000 | |
| 6 | Đế/chân máy + ốc + keo dán | Lắp đặt | ~50.000 | |
| | **TỔNG phần mua thêm** | | **≈1.010.000** | Rẻ hơn 1 máy ATP y tế hàng chục lần — điểm "tính cộng đồng" |

## 3. LẮP ĐẶT CHUẨN (đưa vào báo cáo + demo)

1. **Cụm cảm biến** (C270 + LD2450) gắn cạnh nhau trên vỏ, cao ~1,4–1,6 m,
   hướng vào khu vực người bệnh (ghế/ngủ/bàn).
2. Radar: đặt ngang, mặt anten hướng người; **không** có kim loại chắn trước anten;
   khoảng cách làm việc 0,75–6 m (thông số LD2450).
3. Camera: kiểm tra hình không ngược sáng (cửa sổ sau lưng người bệnh = tránh);
   nền tĩnh giúp YOLO-pose ổn định.
4. Laptop: cắm nguồn, tắt sleep, tắt Windows Update tự động (checklist nghiệm thu S0).
5. Phần mềm: chạy `venv`/python + `streamlit run app_family.py`; chọn COM port
   radar thật trong tab 📡 Radar (không còn cứng COM3 — SYS-17).

## 4. HÌNH ẢNH BẮT BUỘC CHO BÁO CÁO/POSTER (chụp khi lắp 08–09/09)

- [ ] Ảnh cụm C270+LD2450 trên vỏ (gần + xa)
- [ ] Ảnh sơ đồ kết nối chụp màn hình Device Manager (COM port radar)
- [ ] Ảnh app chạy thật: tab 📷 NORMAL → tab 📊 NIHSS → tab 🏥 QR/PDF
- [ ] Ảnh tab 📡 Radar LIVE khi có người đi/ngã nệm
- [ ] Ảnh `nvidia-smi` + benchmark hiệu năng (prototype_benchmark.py) — chứng minh GPU dùng thật
- [ ] Video 30s lắp đặt (tái sử dụng cắt ghép vào video <3 phút)

## 5. THÔNG SỐ KỸ THUẬT ĐƯA VÀO BÁO CÁO (benchmark thật 08/09/2026)

| Chỉ số | Giá trị | Nguồn |
|---|---|---|
| Độ trễ chu kỳ phân tích (face+arm+gait+fusion) | **≈81 ms** / chu kỳ app 3000 ms → dư 2919 ms | `prototype_benchmark_20260908_050756.json` |
| Chi tiết ms (mean): face 15.3 · arm 27.7 · gait-CPU 17.4 · gait-CUDA 20.5 · fusion ~0 | GPU không nhanh hơn ở model nhỏ 640×480 (overhead transfer) — công bố trung thực | benchmark |
| Camera C270 thật | **1280×720 @ 32.2 fps** | benchmark |
| Thời gian phân tích radar 1 nhịp | 600 ms (0,6 s @3 Hz) | thiết kế |
| Chu kỳ cảnh báo đầu-cuối (dấu hiệu → còi + tin nhắn) | [ĐIỀN: S2 nghiệm thu bấm giờ] | S2 |
| Bộ nhớ GPU chiếm dụng khi chạy đủ 5 module | [ĐIỀN sau khi app chạy CUDA] | benchmark |
| Chi phí phần cứng mua thêm | ≈1,01 triệu | BOM |
| Môi trường chạy đã chuẩn | torch 2.14.0+cu126 · torchvision 0.29.0+cu126 · numpy 2.3.4 (pin! 2.5 làm numba/librosa gãy) | 08/09 |

## 6. RỦI RO + AN TOÀN (bắt buộc trong báo cáo — KH "xác định rủi ro tiềm năng")

| Rủi ro | Giảm thiểu |
|---|---|
| Ngã khi kiểm chứng radar | chỉ ngã lên nệm/gối dày, có người bảo hộ (M5-01) |
| Camera thu hình người cao tuổi | xử lý tại máy cục bộ, KHÔNG upload internet; radar thay camera ở nhà vệ sinh/ngủ = bảo vệ riêng tư |
| Sai lệch chẩn đoán | disclaimer KHÔNG PHẢN CHẨN ĐOÁN + GỌI 115 ở mọi màn hình; NIHSS là ước tính ± CI |
| Mất điện/mạng | hệ thống chạy local không cần mạng (trừ bản đồ bệnh viện tĩnh); khuyến nghị nguồn dự phòng |
| Laptop cũ nóng khi chạy 24/7 | giới hạn chu kỳ 3s, GPU chỉ chạy YOLO; benchmark đo nhiệt |

## 7. CÔNG BỐ TRUNG THỰC TRONG BÁO CÁO
- RTX 3050 + C270 là **thiết bị thương mại có sẵn**; phần "chế tạo" của đội =
  (1) kiến trúc 4-Layer Defense + fusion 5 modal, (2) phần mềm PSCS, (3) quy trình
  lắp đặt + kiểm chứng (benchmark, kịch bản nghiệm thu), (4) phát hiện khoa học
  M1-06 (tín hiệu bất đối xứng mặt trong dataset công khai, AUC 0.845 block-CV).
- Radar LD2450 là module thương mại — đội tự viết driver đọc khung UART @256000
  (radar_module.py) + thuật toán ngã (displacement + inactivity + audio-gate).
