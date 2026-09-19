# CHECKLIST CHUẨN BỊ — PHẦN VIỆC CỦA ĐỘI (đến nộp hồ sơ 05/10/2026)

> Đánh dấu [x] khi xong. Mọi kết quả chạy được báo lại để AI chạy eval + ghi sổ nhật ký.

---

## 0️⃣ HÔM NAY — KHẨN (rủi ro tuân thủ)

- [ ] **GVHD ký phê duyệt kế hoạch nghiên cứu** (KH 3.2 — ký TRƯỚC khi nghiên
  cứu, không có = KHÔNG ĐƯỢC CHẤM). Nếu chưa có: in kế hoạch + mời GVHD ký ngay.
- [ ] Kiểm tra **hạnh kiểm Khá+** 2025–2026 của mỗi thành viên, **mỗi em 1 dự án**

---

## 1️⃣ MUA SẮM (cuối tuần 13–14/09, tổng ≈1.010.000đ)

- [ ] **LD2450 + board USB-TTL (CP2102/CH340)** ≈450k — hỏi chủ shop: board 24GHz
  mmWave, UART 5V; **đúng baud 256000**
- [ ] **Logitech C270** ≈350k (nếu chưa có webcam rời)
- [ ] Vỏ/hộp gắn tường: hộp gỗ nhựa ≈100k (kích thước ≥ 10×6×4cm đủ chứa C270 + LD2450)
- [ ] Cáp USB nữ→nam nối dài ×2 ≈60k (3–5m)
- [ ] Đế/chân máy + ốc vít + keo dán ≈50k

---

## 2️⃣ QUAY VIDEO E3 ARM (cuối tuần, trước 18/09)

**Thiết bị:** webcam C270 hoặc điện thoại (720p+), dựng NGANG, cách người ~1.5–2m,
thấy trọn ĐẦU + HAI TAY, ánh sáng đủ, nền tĩnh.

**Người:** 3–5 thành viên/người nhà (in biểu mẫu tự nguyện 1 trang, mỗi người ký).

**Mỗi người quay 2 clip, mỗi clip ~30s (10s chuẩn bị + 20s hành động):**

| Thư mục | Hành động | Ý nghĩa |
|---|---|---|
| `Golden_Watch_KHKT/thu_tuc/` | Ngồi/đứng, GIƠ 2 TAY NGANG 90°, GIỮ YÊN 20s | bình thường (label 0) |
| `Golden_Watch_KHKT/tha_tay/` | Giơ 2 tay ngang → **TỰ THẢ RƠI 1 TAY** (đổi bên giữa các người), giữ tay buông 15s | mô phỏng drift NIHSS item5 (label 1) |

- [ ] Tên file: `ten_nguoi_01.mp4` (VD `an_01.mp4`, `an_02.mp4`) — ghi ai thả tay bên nào
- [ ] Cả 2 clip cùng người quay CÙNG điều kiện (đèn, vị trí)
- [ ] Xong → báo AI chạy `training/eval_arm_e3.py` → NK-18

---

## 3️⃣ RADAR LD2450 (cuối tuần – đầu tuần, trước 20/09)

**Nối dây (cắt nguồn khi nối!):**
1. LD2450: chân **5V → 5V board, GND → GND, TX radar → RX board, RX radar → TX board**
2. Cắm USB vào laptop → Device Manager (Win+X) → Ports (COM & LPT) → ghi COM số mấy
3. Test: `PYTHONUTF8=1 python module5/test_radar_hardware.py --port COM5`
   (lỗi baud → kiểm tra lại 256000; không thấy tín hiệu → đổi chéo TX/RX)

**10 kịch bản (mỗi kịch bản 1–2 phút, có người quan sát ghi thời điểm):**
- [ ] 1. Đi bộ qua lại trong tầm 1–6m
- [ ] 2. Ngã lên NỆM DÀY (⚠️ CÓ NGƯỜI BẢO HỘ, không ngã sàn)
- [ ] 3. Ngồi bất động 2 phút
- [ ] 4. Đứng rồi từ từ ngồi xuống
- [ ] 5. Hai người cùng phòng đi lại
- [ ] 6. Ra khỏi phòng (rãnh radar trống)
- [ ] 7. Vẫy tay chỉ (chuyển động cục bộ)
- [ ] 8. Lcrawler/bò trên nệm (kích thước vật thể thấp)
- [ ] 9. Quay lưng radar (góc ±80° biên)
- [ ] 10. Đi lại rất chậm (mô phỏng người yếu)

- Xong → báo AI + COM port → chạy M5-02 + trụ cột D → NK-19

---

## 4️⃣ GHI ÂM E2 TIẾNG VIỆT (22–26/09)

**In 3 câu chuẩn này (cùng câu cho MỌI người):**
1. "Bố tôi mùa này hay đọc báo bên bậu cửa."
2. "Sáng nay sương sớm phủ kín sân trường."
3. "Em sinh ngày hai mươi ba tháng chín năm hai nghìn không trăm linh tám."

**Người:** ≥10 người (ưu tiên người cao tuổi + người thân, in biểu mẫu tự nguyện).

**Cách ghi (điện thoại, phòng yên, miệng cách máy 20–30cm):**
- [ ] File 1 — `vn_thuong/ten_tuoi_01.wav`: đọc 3 câu TỰ NHIÊN, chậm rõ
- [ ] File 2 — `vn_liu/ten_tuoi_01.wav`: đọc CÙNG 3 câu với **LƯỠI CẮN NHẸ/đọc méo
  tiếng** (mô phỏng nói líu đột quỵ)
- mp3/m4a cũng được (script tự chuyển) — tên file ghi tuổi: `ba_68_01.wav`
- Xong → báo AI chạy `training/eval_speech_vn_e2.py` → NK-20

---

## 5️⃣ SYS-18 NIHSS — 50 VIDEO + 2 CHẤM VIÊN (19–28/09)

- [ ] Tìm video trên YouTube: kênh *NIHSS training/certification* (American
  Stroke Association, "NIH stroke scale assessment"), ưu tiên video CÓ điểm
  NIHSS công bố theo từng bệnh nhân — tải về bằng công cụ tải video hợp pháp
- [ ] Mục tiêu ~50 video, ghi URL + điểm công bố vào cột `ref_total` của
  `NIHSS_SY18_bang_cham_video.csv`
- [ ] **2 CHẤM VIÊN ĐỘC LẬP** xem từng video, điền `rater1_item4/5/6/10` và
  `rater2_item4/5/6/10` (0–2 / 0–4 / 0–4 / 0–2), sau đó thống nhất điền `agreed_*`
- [ ] Xong → báo AI: chạy `nihss_video_toolkit.py score-video` từng video +
  `eval` → Bland–Altman/wκ → NK-21

---

## 6️⃣ PROTOCOL B — 100 TÌNH HUỐNG NGƯỜI KHỎE (21–23/09, cần lịch đội)

Chạy: `PYTHONUTF8=1 python training/protocol_b_defense_ablation.py capture`
(gắn nhãn kịch bản bằng phím theo hướng dẫn trên màn hình; mỗi chu kỳ 2s,
mỗi tình huống ~30s).

Phân bổ 100 tình huống:
- [ ] 25 cười / nói to tự nhiên
- [ ] 15 ngáp
- [ ] 15 quay nghiêng / che mặt thoáng
- [ ] 15 ánh sáng yếu / đèn vàng
- [ ] 15 đeo khẩu trang
- [ ] 15 đi chậm, tư thế người cao tuổi

- Xong → `eval` → defense ablation 4 lớp + replay early-warning + abstention
  trên data THẬT → NK-22 (thay số simulated của NK-11)

---

## 7️⃣ CHẠY 72H LIÊN TỤC (25/09–03/10)

- [ ] Laptop cắm nguồn, đặt ở lab/hộ nhà thành viên, camera+radar hướng vị trí sinh hoạt
- [ ] Tắt sleep: Settings → System → Power → Screen and sleep → **Never**
- [ ] Tắt Windows Update tự động (dừng dịch vụ wuauserv trong 3 ngày)
- [ ] Chạy `PYTHONUTF8=1 python web_server.py` — ghi giờ bắt đầu
- [ ] Ghi sổ: mọi lần crash/cúp điện/thao tác, thời điểm; chụp màn hình dashboard hằng ngày
- [ ] Xong → AI tổng hợp FAR/24h + uptime → NK-23

---

## 8️⃣ HỒ SƠ DỰ THI (01–04/10) ⚠️ NỘP 05–10/10

- [ ] **Báo cáo ≤15 trang** TNR14, lề 3/2/2/2, KHÔNG tên đơn vị (AI viết từ dàn ý
  `BAO_CAO_DU_AN.md` + số `TONG_KET_NGHIEN_CUU.md`; người: duyệt + điền tên đội/GVHD)
- [ ] **Video <3 phút**: quay màn hình app thật (NORMAL → dấu hiệu → EMERGENCY +
  QR) + cận cảnh cụm cảm biến; kịch bản 3 hồi, không diễn viên bệnh nhân
- [ ] **Poster PL3** (theo mẫu Phụ lục 3)
- [ ] **Sổ nhật ký PL2**: chép `so_nhat_ky.md` (NK-01→) vào sổ in + chữ ký
- [ ] **PL1 khai báo AI**: hoàn thiện + ký (`PL1_khai_bao_su_dung_AI.md`)
- [ ] **Khai báo kế thừa** nếu dùng kết quả fga_project (nêu rõ nguồn dataset)
- [ ] **Prototype vật lý**: gọn LD2450 + C270 vào vỏ, test lại S1–S4
- [ ] Chụp đủ ảnh bắt buộc (PROTOTYPE_SPEC.md §4): cụm cảm biến, COM port,
  app từng tab, radar LIVE, nvidia-smi
- [ ] Repo Git chuyển PRIVATE + in hồ sơ giấy

---

**Khi bạn báo "xong mục nào" → AI chạy eval + ghi NK-xx tương ứng trong ngày.**
