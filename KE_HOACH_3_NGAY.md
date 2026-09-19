# KE_HOACH_3_NGAY.md — KẾ HOẠCH 3 NGÀY HOÀN THIỆN HỒ SƠ (deadline nộp cấp trường 03/10/2026)

> Mục tiêu: 3 ngày làm lõi (18–20/09) + buffer 21/09→02/10 để sửa/kiểm tra,
> nộp trước 03/10. Hồ sơ gồm 5 món (theo quy chế):
> **sổ nhật ký (PL2) + báo cáo ≤15 trang + video <3 phút + khai báo AI (PL1)
> + poster online (PL3)**.

## Nguyên tắc sắt (đọc trước khi làm bất cứ việc gì)

1. **KHÔNG train lại model, KHÔNG đổi ngưỡng, KHÔNG đổi code chấm điểm.**
   Model đóng băng (NK-12): face `face_blend_v3_full_20260912_181937.json`,
   speech production giữ nguyên, gait rule-based. Số công bố = số protocol đã
   khóa — mọi số mới chỉ là **kiểm chứng ngoài**, ghi đè số cũ thì phải nói rõ.
2. **KHÔNG đòi 100% chính xác.** Hệ báo "nghi ngờ + độ tin cậy", không phải
   chẩn đoán. Câu trả lời phỏng vấn chuẩn: "AUC 0.94 ± 0.01 (10 seed),
   thiếu sót đã đo và khai báo".
3. Mọi thay đổi dù nhỏ nhất → ghi entry NK-xx vào `so_nhat_ky.md` cùng ngày.
4. Mọi số đưa vào báo cáo phải truy vết được về JSON/file kết quả
   (đã đối chiếu hết NK-26 — không số nào bịa).

## Bước 0 — Làm NGAY hôm nay (17/09, trước khi sang Ngày 1)

- [ ] **Git commit** toàn bộ đồ hiện có (~1,100 dòng chưa commit: L-35 fix,
      vosk, mesh HUD, DU_CU_KHOA_HOC, kit này) — chốt "bản an toàn".
- [ ] In **20 bản** `MAU_DONG_Y_SU_DUNG_DU_LIEU.md` (mỗi người 1 tờ, ký trước khi quay).
- [ ] In/mở sẵn 2 CSV: `data/thu_tu_lieu/CSV_DANH_MUC_DU_LIEU.csv` (điền khi thu)
      và `CSV_NIHSS_2RATER.csv` (điền khi chấm).
- [ ] Kiểm tra thiết bị: laptop RTX3050 sạc đầy, webcam C270, radar LD2450
      + cáp, tripod/giá đỡ, nguồn phát dự phòng.
- [ ] Bật `web_server.py` thử 5 phút trước ngày thu (camera + radar + HUD
      vẽ mesh/xương + banner lệch nhẹ arm) — smoke NK-28 đã PASS nhưng
      phải thấy bằng mắt trên máy THẬT trước giờ thu.

## NGÀY 1 — 18/09 (THỨ SÁU): THU DỮ LIỆU

**Sáng (8h–11h30): chuẩn bị + face**
1. Dọn 1 phòng đủ sáng, treo webcam ngang tầm mặt.
2. Mời 10–15 người (gia đình + bạn lớp), mỗi người: đọc+ký đơn đồng ý
   → nhận mã P01, P02…
3. Face mỗi người 3 tình huống × ~10 giây qua `web_server.py` (quay màn
   hình hoặc dùng chức năng ghi nếu có; tối thiểu chụp khung tĩnh):
   - bình thường nhìn camera 10s
   - cười tự nhiên 10s
   - giả lập méo mặt (lạm phát má/méo miệng một bên) 10s
   → lưu `data/thu_tu_lieu/face/P{XX}_{tinh_huong}_fNN.jpg`.

**Chiều (13h–17h): arm + gait + speech (trụ cột E3 + E2)**
4. Arm E3: mỗi người giơ 2 tay vuông góc 10 giây × 2 lần, quay MP4 ngang,
   đủ sáng, có 1 lần tay intentionally tuột xuống (giả lập drift).
5. Gait: đi thẳng 3m, quay ngang ~2–3 lần/người (1 lần đi bình thường,
   1 lần bước cố tình bất đối xứng).
6. Speech E2 (tiếng Việt): mỗi người đọc 2 câu vào micro laptop (ghi WAV):
   - câu 1 bình thường: "Hôm nay trời đẹp, chúng em xin trình bày dự án."
   - câu 2 ngược vị (khó nói hơn): đọc ngược từng từ của câu 1.
   → lưu `data/thu_tu_lieu/speech/P{XX}_doc_cau{N}.wav` (16kHz mono nếu được;
   khác chuẩn thì dùng librosa resample khi eval).

**Tối (19h–21h): radar + gom dữ liệu**
7. Radar 10 kịch bản × 30s (mỗi kịch bản 1 file CSV xuất từ LD2450):
   đi lại bình thường ×3, đứng yên ×2, ngã giả lập (người ngồi đột ngột) ×2,
   vẫy tay ×1, ra vào phòng ×2.
8. Chạy `python training/ingest_video.py thu_muc_video/ --frames 15` cho
   toàn bộ MP4 → khung jpg + wav 16kHz, dán bảng gợi ý cuối output vào
   `CSV_DANH_MUC_DU_LIEU.csv`.
9. Ghi NK-31 vào sổ (số người, số file mỗi loại, sự cố nếu có).

## NGÀY 2 — 19/09 (THỨ BẢY): CHẤM MÙ + EVAL + ĐÓNG GÓI

**Sáng (8h–12h): chấm NIHSS 2 rater (món quan trọng nhất của external validation)**
1. Quay 5–10 video "tổng hợp 1 người 1 phòng" (mặt + giơ tay + nói trong
   1 cú quay liền, ~1 phút/video) → đặt tên `V01.mp4…V{NN}.mp4`.
2. 2 rater (em + 1 bạn/GV) **chấm ĐỘC LẬP, không nhìn nhau**, theo NIHSS
   Ý 4/5/6/10, điền riêng cột `r1` và `r2` trong `CSV_NIHSS_2RATER.csv`
   (chỉ chấm khung jpg trích sẵn + nghe wav — làm được chấm mù thật).
3. Sau khi cả 2 chấm xong mới thống nhất cột `agreed`; chỗ lệch nhau
   KHÔNG sửa — giữ nguyên để tính κ (đó là bằng chứng độ tin cậy).

**Chiều (13h–17h): chạy eval + khởi động 72h**
4. Chạy bộ eval theo số NK đã giữ chỗ:
   - E3 arm: NK-18 (`training/eval_arm_e3.py`)
   - E2 speech tiếng Việt: NK-20 (`training/eval_speech_vn_e2.py`)
   - NIHSS: NK-21 (`training/nihss_video_toolkit.py eval` → MAE/RMSE/κ/
     Bland–Altman so điểm máy vs `agreed`)
   - radar: NK-19 (10 kịch bản → có/không cảnh báo đúng)
5. **Bắt đầu chạy 72h liên tục** (hoặc 24h nếu chưa kịp — trung thực ghi
   "phiên chạy rút gọn") của web_server trong phòng lab/nhà; chụp màn hình
   dashboard mỗi sáng. Ghi NK-22.
6. Đóng gói vật lý sản phẩm: dán nhãn module, dây gọn, **chụp ảnh từng
   bộ phận cho BOM** (tổng 1,010,000đ — cần ảnh cho báo cáo mục Chế tạo).

**Tối: backup 3 nơi** (laptop + USB + Google Drive). Chép các số eval mới
vào sổ giấy.

## NGÀY 3 — 20/09 (CHỦ NHẬT): LÀM HỒ SƠ (5 MÓN)

1. **Báo cáo 15 trang** — khung có sẵn, ghép từ tài liệu đã viết:
   - Vấn đề + Thiết kế & Phương pháp: lấy từ `DU_CU_KHOA_HOC.md` mục 1–2
     (NIHSS/FAST + y văn + bảng ánh xạ module).
   - Chế tạo & kiểm tra: BOM + ảnh đóng gói (Ngày 2) + bảng kiểm định
     `BANG_THANG_DO_KIEM_DINH.md` + số eval mới NK-18→21.
   - Sáng tạo: 3 lớp "chưa từng có" (BlockSweep tự bắt leakage ×2 +
     abstention T pillar A + early warning 20/20@21.6 phút) — `TONG_KET_NGHIEN_CUU.md`.
   - Định dạng: A4, TNR 14, lề 3/2/2/2, **KHÔNG tên trường/đơn vị**.
2. **Video <3 phút** — kịch bản: 0:00–0:30 vấn đề (số liệu đột quỵ VN),
   0:30–2:00 demo thật web_server (mesh mặt + xương tay + cảnh báo +
   radar), 2:00–2:50 số liệu + kiểm định, 2:50–3:00 kết. Quay màn hình
   bằng OBS, GIỌNỮI dùng dữ liệu công khai hoặc đã che mặt.
3. **PL1 khai báo AI**: khai báo trợ lý AI hỗ trợ code/docs + kế thừa
   FAST.AI JAMA Neurology (nhóm [B] — chi tiết xác minh sau 13/10).
4. **PL3 poster online**: dùng lại 13 PNG trong
   `test_results/metrics_pack_20260907_223217/` + ảnh demo NK-28.
5. **Sổ nhật ký giấy**: chép lại từ `so_nhat_ky.md` + `NHAT_KY_DU_AN.md`
   theo PL2 (mỗi ngày 1–3 trang, bìa + mục lục + phụ lục).

## BUFFER 21/09 → 02/10 (không để trống!)

- **Protocol B rút gọn**: 20–30 tình huống thay vì 100 (bảng KICH_BAN_
  NGHIEM_THU S0–S6, chọn quan trọng nhất) → NK-23. Nếu không kịp: khai
  báo trung thực "kế hoạch đã thiết kế, chạy phiên bản rút gọn".
- Gửi báo cáo nháp cho GVHD góp ý (mỗi GV 1 dự án — chốt giấy ký kế hoạch
  TRƯỚC ngày nghiên cứu nếu chưa ký!).
- Kiểm tra chéo hồ sơ theo checklist quy chế (đúng 5 món, đúng định dạng).
- NỘP SỚM (mục tiêu 30/09–01/10), đừng đợi đúng 03/10.

## Những gì CHỐT KHÔNG làm trong 3 ngày (tránh sa lầy)

- Không retrain face/speech v4, không thêm wav2vec2/openSMILE vào production.
- Không thu thêm dataset ngoài (v2.0 chốt 10/09: chỉ dùng sẵn có + tự thu).
- Không sửa fusion/defense engine (trừ bug chặn hiển thị — phải ghi NK-xx).
- Không hứa số chưa có protocol trong báo cáo — ghi rõ "kết quả sơ bộ".
