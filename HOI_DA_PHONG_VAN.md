# BỘ CÂU HỎI PHỎNG VẤN — GOLDEN-WATCH (30 câu, trả lời theo số thật)

> Cách dùng: ĐỌC HIỂU, đừng học thuộc. Mỗi câu trả lời 20–40 giây.
> Nguyên tắc vàng: không biết → "Em chưa đo kiểm chứng điều đó, nhưng từ số
> hiện có em suy luận…" — KHÔNG bịa số. Mọi số dưới đây đều có tệp trong
> test_results/ để chỉ ra khi được hỏi.

## A. Ý TƯỞNG & BỐI CẢNH

**A1. Vì sao em chọn đột quỵ mà không phải bệnh khác phổ biến hơn?**
Đột quỵ là bệnh có "khoảng vàng" cực kỳ cụ thể: 4.5 giờ đầu để dùng thuốc
thrombolysis, mỗi phút chậm hàng triệu neuron mất. Vấn đề không phải thiếu
bệnh viện mà là **người nhà không nhận ra và chậm gọi 115** — Việt Nam chỉ
khoảng 6–10% bệnh nhân đến trong khoảng vàng. Hệ thống của em đánh đúng vào
điểm nghẽn đó: sàng lọc TẠI NHÀ, trước bệnh viện (PSCS).

**A2. Sản phẩm này chẩn đoán đột quỵ phải không?**
KHÔNG và em nói rõ: đây là **sàng lọc hỗ trợ, KHÔNG phải chẩn đoán**. Trên
mọi màn hình và báo cáo đều ghi "ƯỚC TÍNH HỖ TRỢ — bác sĩ chấm NIHSS chuẩn".
Sản phẩm chỉ làm 2 việc: phát hiện sớm dấu hiệu để **gọi 115 nhanh hơn**, và
gửi thông tin ban đầu (QR/PDF) cho tuyến sau.

**A3. Điểm khác của em so với đồng hồ/watch y tế bán trên thị trường?**
Watch đo được 1–2 chỉ số sinh hiệu. Em đo **5 kênh dấu hiệu thần kinh** đúng
theo thang NIHSS (méo mặt, nói líu, yếu tay, dáng đi, chuyển động qua radar)
và hợp nhất thành kết luận có kiểm định thống kê — thứ không thiết bị tiêu
dùng nào làm. Chi phí cũng khác một bậc: bộ prototype ~1,01 triệu đồng.

**A4. "PSCS" nghĩa là gì?**
Prehospital Stroke Clinical Screening — sàng lọc lâm sàng giai đoạn trước
bệnh viện. Em dùng khái niệm này để nhấn: mô hình nhắm vào bối cảnh gia đình,
không nhắm vào phòng khám.

## B. KIẾN TRÚC & VẬN HÀNH

**B1. Một chu kỳ phân tích diễn ra như thế nào?**
3 luồng song song: camera 30fps chỉ để hiển thị; cứ 2 giây hệ thống lấy khung
phân tích qua 4 module (mặt MediaPipe+ML, tay YOLO-pose, dáng đi YOLO-pose,
giọng nói MLP); radar LD2450 chạy luồng riêng. Kết quả đi qua DefenseEngine
4 lớp → FusionEngine → ước tính NIHSS 4 item → phân loại → cảnh báo. Toàn bộ
chu kỳ phân tích mất **~81 ms** (đo benchmark) — camera không bao giờ bị khựng.

**B2. Mỗi module hoạt động ra sao?**
- **Mặt:** 478 landmark MediaPipe → 28 đặc trưng (5 tỷ lệ + 20 blendshape
  chênh trái–phải + 3 góc pose) → hồi quy logistic.
- **Tay:** YOLOv8n-pose 17 keypoints → tỷ lệ thả tay so với tư thế chuẩn.
- **Dáng đi:** YOLO-pose 6 chỉ số dưới cơ thể, rule-based.
- **Giọng nói:** 48 đặc trưng (MFCC 13×3 + pitch + năng lượng) → MLP
  48→256→128→64.

**B3. Tại sao cần radar LD2450 khi đã có camera?**
Camera có 2 điểm mù: **đêm/ánh sáng yếu** và **vấn đề riêng tư** (phòng ngủ).
Radar 24 GHz đo chuyển động vi (thở, té ngã, bất động bất thường) không cần
ánh sáng, không lưu hình ảnh. Trọng số fusion: radar 15%.

**B4. Trọng số hợp nhất lấy ở đâu ra, sao không phải 50/50?**
Tra số theo **độ tin cậy đã đo của từng kênh**: tay 0.30 · mặt 0.20 · nói
0.20 · dáng đi 0.15 · radar 0.15. Kênh nào kiểm định càng tốt, trọng số càng
cao. Khi thiếu module, hệ **chuẩn hoá lại trọng số** thay vì trả về 0 — nên
đêm không có hình vẫn còn radar + nói chấm được.

**B5. 4 lớp Defense là gì, có phải phình to không?**
Là 4 bộ lọc giảm báo giả, mỗi lớp giải 1 loại nhiễu thật:
L1 tự hiệu chuẩn theo chính người dùng (mặt/gait/di chuyển nền tảng), L2 nhận
biết ngữ cảnh (cười, ngáp, nhiều người) — L2 chỉ gắn nhãn KHÔNG sửa điểm vì
bài học L-08: ngữ cảnh không được che đột quỵ thật; L3 thời gian — dấu hiệu
phải kéo dài 30s mới lên điểm đủ (chống vụt qua khung hình); L4 sàn thích
nghi trong vùng xám [30,40) trừ 5 điểm rồi chuyển thành "theo dõi thêm".

**B6. Nếu cúp điện/ngắt mạng thì sao?**
Toàn bộ chạy **offline trên máy cục bộ** — không gọi cloud, không cần mạng
để phát hiện. Web server chỉ phục vụ xem từ máy khác trong nhà qua
127.0.0.1/cổng 5001. Cúp điện = mất giám sát (hạn chế thật, em ghi rõ).

**B7. Vì sao em tự dựng web server FastAPI mà không dùng Streamlit?**
Streamlit chạy lại toàn bộ script mỗi tương tác → video giám sát giật.
FastAPI + MJPEG cho **~15–20 fps ổn định**, mở được bằng VLC, 3 thread tách
biệt (camera / radar / phân tích) — quyết định kỹ thuật em rút ra sau khi
thử thất bại với Streamlit (nhật ký L-30).

## C. KIỂM ĐỊNH & SỐ LIỆU (nhóm dễ bị "soi" nhất)

**C1. Con số chính của em là gì?**
- Mặt: **AUC 0.94 ± 0.01** (10 phép chia ngẫu nhiên khác nhau), độ nhạy
  ~82%, độ đặc hiệu ~92%.
- Dáng đi (thảm lực PhysioNet): AUC **0.879** LOSO.
- Giọng nói: người thật leave-one-speaker-out **AUC 0.62** (tự trích 48
  đặc trưng) — **0.663** với bộ 88 đặc trưng chuẩn openSMILE.
- 68 test tự động PASS (58 + 10) mỗi lần thay code.

**C2. Nghe nói mô hình của em từng đạt 100%? Sao không khoe?**
Đúng — và đó là một trong 2 phát hiện em tự hào nhất. Mô hình cây (HistGB)
đạt AUC **1.000** trên dataset mặt, nhưng em nghi ngờ và dựng **Giao thức
BlockSweep**: quét nhiều kích thước block dữ liệu để kiểm tra. Kết quả: cây
≈1.000 ở MỌI kích thước block (dù chỉ còn 8 block) = nó **học thuộc từng
người**, không học dấu hiệu bệnh. Mô hình tuyến tính giữ 0.91–0.94 ổn định.
Chúng tôi coi AUC 1.000 là **chỉ số lỗi**, không phải thành tích.

**C3. Giọng nói 0.992 tụt xuống 0.62 — dự án thất bại phần đó?**
Đây là phát hiện leakage thứ 2. Số 0.992 cũ chia dữ liệu theo **phiên thu ×
micro** — cùng một người nằm cả tập train lẫn test (thu 2 micro). Khi em
chia lại chuẩn theo **người thật** (LOSO 15 người, protocol ghi trước), AUC
thật là 0.62. Em KHÔNG giấu: báo cáo công bố cả 2 số và rút bài học phương
pháp. Câu chuyện này chính là bằng chứng quy trình khoa học của nhóm hoạt
động — phát hiện lỗi của chính mình.

**C4. Ai bảo 0.62 mới đúng mà 0.992 sai?**
Nguyên tắc dữ liệu y tế: đánh giá mô hình phải trả lời "người MỚI thì sao?"
— toàn bộ mục đích là chấm người chưa từng thấy. LOSO (leave-one-speaker-
out) chính là làm điều đó 15 lần. 0.992 trả lời câu hỏi khác: "file mới của
người CŨ thì sao" — câu hỏi không có giá trị ứng dụng.

**C5. Vì sao mô hình mặt chọn hồi quy logistic mà không deep learning?**
3 lý do đo được: (1) tại dataset này, mô hình phức tạp bị leakage người
(C2); (2) LogReg **ổn định 0.91–0.94 ở mọi kích thước block và mọi seed** —
trả lời được "chia khác thì sao?"; (3) chạy on-device: 28 phép nhân — gọn,
minh bạch, giải thích được hệ số từng đặc trưng. Bài học: đầu dự đoán PHẢI
đơn giản khi dữ liệu nhỏ.

**C6. openSMILE 88 đặc trưng chuẩn quốc tế hơn 48 đặc trưng của em — sao
không đổi?**
Em đã so sánh CÙNG file, CÙNG fold: 0.663 vs 0.620, **+0.043**. Cải thiện
nhỏ, không đáng đánh đổi thêm phụ thuộc thư viện khi chạy tại nhà. Báo cáo
công bố cả hai — đây là nghiên cứu đặc trưng có kiểm soát, không phải tự ý
giữ cái của mình.

**C7. Ngưỡng cảnh báo đặt sao, cứ 80 điểm là còi hả?**
Cảnh báo khẩn khi điểm hợp nhất ≥80 hoặc quy tắc FAST: ≥2/3 dấu hiệu với
điểm ≥50. Sau cảnh báo có **cooldown 60 giây** tránh còi liên tục. Dưới 50
chỉ hiển thị "theo dõi". Toàn bộ ngưỡng nêu rõ trong tài liệu, không giấu.

**C8. NIHSS của máy tính ra sao, so bác sĩ được không?**
Máy ước tính 4/15 item NIHSS (mặt 0–3, tay 0–4, chân 0–4, nói 0–2) = tối đa
**13 điểm**, kèm khoảng tin cậy Monte Carlo 1000 lần lặp. Đang xây dựng
protocol so với 50 video có bác sĩ chấm chuẩn công khai (MAE, Bland–Altman,
weighted-κ) — toolkit đã xong, đang thu video.

**C9. Wilson CI là gì, sao em nhấn nhiều?**
Tỷ lệ % tính từ n nhỏ sẽ lừa nếu không có khoảng tin cậy. Wilson cho khoảng
đúng đắn ở n nhỏ — ví dụ độ nhạy 82% trên 3715 ảnh có khoảng hẹp, nhưng số
n nhỏ (E3/E2) em đều kèm khoảng và ghi rõ "smoke test n nhỏ".

**C10. Mô hình gait 0.879 nối vào camera chưa? Tại sao chưa?**
CHƯA — và em nói thẳng: model 0.879 học trên **8 đặc trưng thảm lực**
PhysioNet, không tương thích trực tiếp 6 chỉ số từ camera. Gán số 0.879 cho
camera là khai sai — em không làm. Camera gait hiện chạy rule-based; đây là
hạn chế đã ghi, là hướng phát triển.

## D. ĐẠO ĐỨC, HẠN CHẾ, AN TOÀN

**D1. Rủi ro lớn nhất của sản phẩm? Người nhà quá tin vào máy?**
Đúng — đó là lý do mọi đầu ra đều ghi "KHÔNG PHẢN CHẨN ĐOÁN — GỌI 115" và
hệ thống được thiết kế theo hướng **lệch về báo động** hơn là im lặng (độ
đặc hiệu chấp nhận thấp hơn để giữ độ nhạy). Trường hợp nghi, kết luận luôn
là "theo dõi và gọi 115 khi nghi", không bao giờ là "bình thường, không lo".

**D2. Báo giả nhiều thì sao? Người ta tắt luôn hệ thống.**
Đây là lý do kiến trúc 4 lớp L1–L4 tồn tại: đo kiểm smoke cho thấy tắt hết
phòng vệ → ~18% chu kỳ khỏe báo động; bật đủ → 0%. Protocol thực địa 5 ngày
tại nhà đang đo "báo giả/24h" như chỉ số chính. Em không tuyên bố số thiết
kế — em sẽ trình số đo được.

**D3. Dữ liệu người dùng đi đâu? Có camera quay 24/7 thì riêng tư sao?**
Hoàn toàn cục bộ: không cloud, không upload. Chế độ học cá nhân lưu **median
đặc trưng, không lưu ảnh**. Radar không tạo hình ảnh. Video chỉ hiện trên
mạng nội bộ 127.0.0.1.

**D4. Hạn chế lớn nhất em thừa nhận là gì? (câu 100% bị hỏi)**
4 hạn chế, khai đủ: (1) giọng nói người-level còn yếu 0.62 và chưa có corpus
tiếng Việt — AUC tiếng Việt zero-shot đang đo; (2) arm hiện eval n nhỏ, tự
quay; (3) NIHSS máy chưa có đối chuẩn bác sĩ — đang bổ sung 50 video; (4)
chưa kiểm định thực địa rộng — field study 1 hộ, ghi rõ "single-site".

**D5. Vì sao không thử trên bệnh nhân thật?**
Vượt khả năng một dự án học sinh về giấy phép, đạo đức nghiên cứu và an toàn
bệnh nhân. Thay vào đó em hoàn thành đủ **3 bậc kiểm định** chuẩn TRIPOD+AI
(BMJ 2024): nội bộ có leakage-audit → dữ liệu ngoài (video NIHSS, tiếng Việt)
→ thực địa tại nhà. Đó là chuẩn kiểm định y tế quốc tế, và phần lớn mô hình
AI y tế công bố còn dừng ở bậc 1.

## E. TRIỂN KHAI, CHI PHÍ, MỞ RỘNG

**E1. Chi phí sản phẩm?**
Prototype: laptop RTX 3050 (nhiều nhà đã có) + webcam C270 (~200k) + radar
LD2450 (~350k) + vỏ in/hộp (~100k) ≈ **1,01 triệu** chưa tính laptop. Điểm
cộng: rẻ, dân có thể lắp — đúng tinh thần cộng đồng.

**E2. Muốn nhân rộng cần gì?**
3 bước thực tế: (1) đủ corpus dysarthria tiếng Việt để sửa kênh nói; (2)
kiểm định đa hộ/đa trung tâm để hết nhãn single-site; (3) gọn phần cứng thành
một hộp + app điện thoại (kiến trúc web server sẵn sàng vì đã tách API).

**E3. Nếu radar/camera hỏng một cái?**
Hệ vẫn chạy — fusion chuẩn hoá lại trọng số khi thiếu module (B4). Chỉ khi
còn duy nhất radar + giọng thì độ chính xác giảm nhưng không chết hoàn toàn.
Đây là thiết kế "suy giảm nhẹ nhàng", đúng bài toán nhà thật.

**E4. AI trong dự án của em — em khai chưa, dùng cái gì?**
Khai đầy đủ trong Phụ lục 1: em dùng Claude Code/GLM làm công cụ viết code
theo chỉ dẫn của em. Em quyết định kiến trúc, duyệt từng bước, trực tiếp thu
dữ liệu và kiểm thử; mọi thay đổi có nhật ký ngày tháng (Phụ lục 2). Em cho
rằng khai báo trung thực và hiểu rõ từng dòng mình trình bày mới là chuẩn
khoa học — ngược lại mới là vi phạm.

**E5. Nếu em được 1 năm nữa và 100 triệu, em làm gì đầu tiên?**
Thứ tự: (1) hợp tác khoa thần kinh để thu corpus dysarthria + video NIHSS
chuẩn Việt Nam — sửa kênh yếu nhất (nói 0.62, L-02); (2) kiểm định prospective
đa hộ 30 ngày; (3) thử nghiệm thêm mô hình wav2vec2 frozen + đầu tuyến tính
cho tiếng nói — nền tảng đã để sẵn.

---

## MẸO TRẢ LỜI 3 KIỂU CÂU "BẪY"

1. *"Số em copy ở đâu ra?"* → Chỉ tệp: test_results/…, seed 42, script có
   trong training/, chạy lại được.
2. *"Tại sao không X?"* (mọi biến thể) → "Em cân nhắc rồi và chọn Y vì Z đo
   được" — luôn nối về 1 số, không trả lời cảm tính.
3. *"Em hay AI làm?"* → Như E4: trung thực, tự tin — em là chủ dự án, AI là
   công cụ; dẫn Phụ lục 1 + sổ nhật ký.
