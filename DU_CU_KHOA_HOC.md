# DU_CU_KHOA_HOC.md — DẪN CHỨNG KHOA HỌC: TẠI SAO SƠ ĐỒ & SỐ LIỆU NÀY CHỈ RA ĐỘT QUỶ

> **Dùng cho:** báo cáo 15 trang (mục Vấn đề + Thiết kế & Phương pháp) và
> trả lời phỏng vấn. Mọi số hệ thống trích trong tài liệu này ĐÃ được đối
> chiếu về JSON gốc (kiểm toán NK-26, 17/09/2026).
> **Quy ước tin cậy trích dẫn:** [A] = bài báo/sách chuẩn, độ tin cậy cao,
> trích được ngay · [B] = đúng hướng nhưng CẦN XÁC MINH DOI trước khi in
> vào báo cáo (hạn mức tra internet của nhóm AI hỗ trợ mở lại 13/10/2026 —
> ghi trong NK-24). KHÔNG có trích dẫn nào bị bịa.

---

## 1. LUẬN ĐIỂM GỐC: SƠ ĐỒ HỆ = BẢN "KHÔNG CHẠM" CỦA THANG ĐIỂM CHUẨN NIHSS + THANG NHANH FAST

### 1.1. Vì sao chọn NIHSS làm xương sống?

**[A] Brott T, et al.** "Measurements of acute cerebral infarction: a
clinical examination scale." *Stroke*. 1989;20(7):864–870.
→ Bài báo SANG LẬP thang NIHSS: điểm số đột quỵ được xác định bằng
**quan sát không xâm lấn** các chỉ dấu vận động – mặt – ngôn ngữ.

**[A] Lyden P.** "Using the National Institutes of Health Stroke Scale:
a call for absolute accuracy." *Cerebrovascular Diseases*. 2007;23(3):167–175.
→ NIHSS là thang được dùng rộng rãi nhất để định mức độ nặng và quyết
định điều trị tan máu đông.

**Sơ đồ hệ ánh sáng 1-1 lên các ý NIHSS (đây là lý do kiến trúc 5 module
KHÔNG phải ghép ngẫu nhiên):**

| Module đo không chạm | Ý NIHSS tương ứng | Dấu hiệu y khoa |
|---|---|---|
| M1 Mặt (478 landmark + ML v3) | **Ý 4 — Liệt mặt** | méo miệng, sụp khóe, lệch nếp mũi-môi |
| M3 Tay (YOLO-pose + ML yếu tay) | **Ý 5 — Vận động tay** | tay rơi xuống (drift), lệch góc L/R |
| M4 Dáng đi (YOLO-pose) | **Ý 6 — Vận động chân / dáng đi** | bất đối xứng bước, loạng choạng |
| M2 Nói (48 đặc trưng âm học + ML) | **Ý 10 — Khó nói (dysarthria)** | nói ngọng, tốc độ/tiểu điều biến đổi |
| M5 Radar 24GHz | (bổ trợ) phát ngã + chuyển động | dấu hiệu đột quỵ khiến ngã/quẹt |

**[A] Goldberger AL, et al.** "PhysioBank, PhysioToolkit, and PhysioNet."
*Circulation*. 2000;101(23):e215–e220. → chuẩn mở để trích dẫn kho dữ
liệu sinh lý (PhysioNet) mà nhóm dùng cho dáng đi.

### 1.2. Vì sao kết hợp 3 dấu hiệu FAST lại trong 1 hệ?

**[B] Harbison J, Hossain O, Dulay L, et al.** "Diagnostic accuracy of
stroke referrals from a general practice..." *Stroke* / và nghiên cứu
kiểm chứng thang **FAST (Face–Arm–Speech–Time)** — *CẦN XÁC MINH đúng
số trang khi có mạng*.
→ Thang FAST được cứu hộ dùng rộng rãi: **méo mặt + yếu tay + nói khó =
3 dấu hiệu tiên đoán đột quỵ mạnh nhất ngoài bệnh viện**. Hệ nhóm là
cách đo 3 dấu hiệu đó TỰ ĐỘNG – LIÊN TỤC – KHÔNG CHẠM, thêm radar.

---

## 2. DẪN CHỨNG THEO TỪNG MODULE — "TẠI SAO ĐO ĐƯỢC NÓ LÀ ĐỘT QUỶ"

### 2.1. M1 MẶT — méo mặt

**Cơ chế bệnh học:** liệt mặt trung tâm sau đột quỵ nao bán cầu: chỉ
bán mặt DƯỚI mất điều khiển vỏ não (đường vỏ-não trán) → khóe miệng
tuột, nếp mũi-môi mất cân đối. Chuẩn chấm là Ý 4 NIHSS.

**Y văn về đo tự động méo mặt:**
- **[A] Lugaresi C, et al.** "MediaPipe: A Framework for Building
  Perception Pipelines." *arXiv:1906.08172*, 2019. → công cụ landmark
  478 điểm chuẩn Google (kèm file mẫu trong dự án), tách đặc trưng hình
  học mặt.
- **[B] Nhóm nghiên cứu 2021–2025 về chấm điểm liệt mặt tự động
  (facial palsy grading) bằng landmark/deep learning** — còn nhiều bài
  (IEEE/npj Digital Medicine), *chốt 2 bài khi có mạng*. Luận điểm dùng:
  "tỉ lệ bất đối xứng miệng/mắt đo từ landmark tương quan với độ liệt
  mặt chấm tay của bác sĩ (House–Brackmann scale)."
- **[B] Nghiên cứu smartphone telestroke phát hiện méo mặt** — đã thấy
  trong rà soát mới nhất: nghiên cứu **FAST.AI trên JAMA Neurology
  (2024–2025)** dùng AI nhận dạng đột quỵ từ video mặt + lời nói.
  *Đây vừa là dẫn chứng vừa là "phải khai báo kế thừa" theo quy chế.*

**Số của hệ (đã khóa protocol, đối chiếu JSON NK-26):**
- LogReg 28 đặc trưng (5 tỉ lệ lâm sàng + 20 bất đối xứng blendshape +
  3 góc đầu), held-out block test: **AUC 0.943 · độ nhạy 80.1% · độ
  chuyên 94.1%** (`models/face_blend_v3_20260909_20260909_201800.json`).
- Ổn định 10 seed: **0.9398 ± 0.0099** — không phải may mắn 1 lần chạy.
- Blendshape làm tăng khả năng phân biệt THẬT: kiểm định DeLong cặp
  **ΔAUC = 0.0143, p = 1.5×10⁻⁸** (NK-24) — trích dẫn phép thử:
  **[A] DeLong ER, DeLong DM, Clarke-Pearce DL.** *Biometrics*.
  1988;44(3):837–845.

### 2.2. M3 TAY — yếu tay / rơi tay (drift)

**Cơ chế:** tổn thương đường tháp (corticospinal) → yếu chi; dấu hiệu
lâm sàng kinh điển là **pronator drift**: giơ 2 tay, tay yếu tụt xuống
và xoay sấp trong vài giây. Đây chính là Ý 5 NIHSS.

**Y văn:**
- **[A] Campbell WW.** *DeJong's The Neurologic Examination* (sách chuẩn
  xét nghiệm thần kinh, nhiều lần tái bản) → mô tả & ý nghĩa drift/
  yếu chi trong đột quỵ.
- **[B] Các nghiên cứu định lượng drift bằng máy ảnh/pose estimation
  sau đột quỵ** — *chốt bài khi có mạng*. Luận điểm: "biến thể độ cao
  cổ tay theo thời gian phân biệt được tay yếu."
- **[B] Patterson KK, et al.** "Gait asymmetry in community-ambulating
  stroke survivors." *Arch Phys Med Rehabil*. 2008 (đối xứng chi sau
  đột quỵ) — dùng chung luận điểm bất đối xứng.

**Số của hệ:** điểm tay = ML 16 đặc trưng (6 khớp + drift + span +
bất đối xứng) + rule dự phòng; ngưỡng WARNING ≥50; **cảnh báo sớm LỆCH
NHẸ đã bổ sung ở NK-28** (lệch góc >10° = đúng bậc cộng điểm +10 trong
bảng rule — bậc này mô phỏng thang drift NIHSS 1–4).

### 2.3. M2 NÓI — khó nói (dysarthria)

**Cơ chế:** đột quỵ vùng vỏ não / tiểu não → điều phối cơ phát âm yếu →
thay đổi pitch (cao độ), pause (ngắt nghỉ), tốc độ nói (WPM), năng
lượng. Chuẩn chấm Ý 10 NIHSS.

**Y văn:**
- **[A] Duffy JR.** *Motor Speech Disorders: Substrates, Differential
  Diagnosis, and Management.* Elsevier (sách chuẩn chuyên khoa) →
  mô tả đặc trưng âm học dysarthria sau đột quỵ.
- **[A] Lowe S, et al.** "Dysarthric speech database for universal
  access research." *Interspeech*. 2016 → **bộ dữ liệu TORGO** (17,635
  file, người thật) — nguồn huấn luyện của hệ.
- **[A] Eyben F, et al.** "The Geneva Minimalistic Acoustic Parameter
  Set (eGeMAPS)..." *IEEE Trans. Affective Computing*. 2016;7(2):190–202
  → bộ 88 đặc trưng âm học CHUẨN quốc tế cho bệnh lý lời nói; hệ dùng
  làm baseline so sánh (openSMILE LOSO 0.66 vs 48-đặc-trưng tự trích
  0.62 — `test_results/speech_opensmile_loso_20260910_221102/`).

**Trung thực bắt buộc (điểm cộng ở vòng Quốc gia):** TORGO gồm người
dysarthria do các nguyên nhân (bại não...), KHÔNG 100% là đột quỵ →
hệ dùng nó làm bài toán đại diện "bất thường cơ nói", và **kiểm chứng
riêng trên tiếng Việt + người thật (E2 — NK-20)**. Số công bố: LOSO
theo NGƯỜI THẬT **AUC 0.62** (LogReg) — số trung thực sau khi tự phát
hiện leakage 0.992 (NK-03); bài học này được khai báo như điểm mạnh
phương pháp luận.

### 2.4. M4 DÁNG ĐI

**Cơ chế:** đột quỵ → bất đối xứng bước, giảm vận tốc, tăng biến thiên
chu kỳ bước; cũng là nguyên nhân ngã.

**Y văn:**
- **[A] Hausdorff JM, et al.** các nghiên cứu biến thiên dáng đi &
  nguy cơ té ngã (PhysioNet Gait in Aging and Disease Database —
  nguồn dữ liệu của hệ, [A] Goldberger 2000 ở trên).
- **[B] Olney SJ, Richards CL** về dáng đi sau đột quỵ — *chốt bài*.

**Trung thực:** dữ liệu PhysioNet gồm (young/elderly/Parkinson) —
KHÔNG thuần đột quỵ; module dùng làm "phát hiện dáng đi bất thường",
giá trị bổ trợ, KHÔNG nối vào camera chấm điểm (quyết định M4-07
trong LOI_SO_MODULE.md — đã khai báo).

### 2.5. M5 RADAR 24GHz (bổ trợ)

**Luận điểm:** radar mmWave đo chuyển động KHÔNG CẦM THIẾT BỊ — người
đột quỵ có thể không đeo/không bấm nút; phát ngã + hoạt động bất
thường là bối cảnh để giảm báo sai và bổ sung cảnh báo khi camera bị
che. **[B] Các nghiên cứu mmWave radar phát ngã & theo dõi sức khỏe
không chạm (IEEE review 2021–2024)** — *chốt 1 review khi có mạng*.

### 2.6. TẦNG VỆ TINH — cảnh báo SỚM (trụ cột B) có y văn không?

CÓ — đây là luận điểm mạnh nhất cho tính "chưa từng có":
- **[A] Johnston SC, Gress DR, Browner WS, Sidney S.** "Short-term
  prognosis after emergency department diagnosis of TIA." *JAMA*.
  2000;284(22):2901–2906 → TIA (đột quỵ thoáng qua) là CẢNH BÁO;
  nguy cơ đột quỵ thật cao ngay sau đó.
- **[A] Rothwell PM, et al.** "A simple score (ABCD) to identify
  individuals at high early risk of stroke after transient ischaemic
  attack." *Lancet*. 2005;366(9479):29–36 (+ ABCD2 ở Lancet 2007)
  → chuẩn quốc tế coi dấu hiệu NHẸ/DẪN-NHỈN trước đó là đủ để can
  thiệp sớm.
→ Hệ đo bằng bằng chứng: mô phỏng drift xấu dần +18/30 phút dưới ngưỡng
50 — hệ cũ KHÔNG BAO GIỜ báo (0/20), hệ có Early Warning báo 20/20
với trung vị **21.6 phút** (`test_results/early_warning_20260912_124947/`,
NK-11). Đó chính là "cảnh báo TIA/prodromal" bằng máy.

---

## 3. TẠI SAO CÁC SỐ CỦA HỆ LÀ BẰNG CHỨNG ĐÁNG TIN (KHÔNG PHẢI BỊA)

1. **Protocol khóa trước khi công bố** (SYS-28 + NK-04): chia block,
   seed 42, model đóng băng; bản fit 100% chỉ là artifact phụ
   (`face_blend_v3_full_*.json` ghi rõ note).
2. **Tự phát hiện và công bố leakage của chính mình** (2 lần — HistGB
   AUC 1.000 ở NK-17/L-34; speech 0.992 session-level ở NK-03/S-1):
   có **Giao thức BlockSweep** chứng minh (`blocksweep_leakage_diagnostic.png`).
   → đây là tư duy đúng theo **[A] Collins GS, et al. "TRIPOD+AI
   statement." *BMJ*. 2024;385:e078378** và **[A] Bossuyt PM, et al.
   "STARD 2015." *BMJ*. 2015;351:h5527** (đã trích trong
   BANG_THANG_DO_KIEM_DINH.md).
3. **Kiểm định thống kê đầy đủ** (NK-24): McNemar chính xác
   (p=0.714 — khác nhau ở ngưỡng, không phải phân biệt), DeLong cặp
   (p=1.5×10⁻⁸), ECE 2.1% (độ hiệu chuẩn), PR-AUC 0.9237 so nền 0.3341.
4. **Kế hoạch kiểm chứng ngoài** (bằng chứng cấp cuối cùng): dữ liệu
   tự thu tiếng Việt + video NIHSS chấm tay 2 rater (NK-18→NK-23) —
   đúng tinh thần TRIPOD+AI: external validation.

---

## 4. BA LUẬN ĐIỂM 30 GIÂY KHI PHỎNG VẤN

1. **"Máy đo đúng thứ bác sĩ đo"** — mỗi module ánh xạ 1 ý NIHSS
   (Brott 1989), gộp lại = FAST tự động (méo mặt + yếu tay + nói khó).
2. **"Số có thể truy vết từng con"** — AUC 0.943 ± 0.01 (10 seed),
   leakage tự phát hiện 2 lần, McNemar/DeLong/ECE đủ bộ (NK-24/26).
3. **"Ngoài phát hiện, còn cảnh báo sớm"** — cơ sở TIA/ABCD2
   (JAMA 2000, Lancet 2005): hệ báo drift xấu dần trung vị 21.6 phút
   trước khi vượt ngưỡng — hệ cũ không bao giờ làm được.

## 5. VIỆC CẦN LÀM SAU 13/10/2026 (khi tra được mạng)
- Chốt 5–6 trích dẫn nhóm [B]: facial palsy grading (2 bài), FAST
  validation, định lượng drift bằng vision, 1 review radar mmWave,
  chi tiết FAST.AI JAMA Neurology (khai báo kế thừa ở Phụ lục 1).
- Kiểm tra lại từng DOI/سال xuất bản trước khi in vào báo cáo.
