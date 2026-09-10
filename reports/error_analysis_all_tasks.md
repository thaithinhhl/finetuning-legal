# Tổng hợp lỗi 3 task — cơ sở để cải thiện data training

Model đánh giá: `Qwen2.5-7B-Instruct` + LoRA (`best_adapter`), BF16, batch_size=1, prompt dùng `instruction` nguyên bản trong file benchmark.

## Kết quả tổng quan

| Task | Nội dung | n | Kết quả | Nhóm data tương ứng |
|---|---|---|---|---|
| 1.2 | Phân loại lĩnh vực (MCQ 6 chọn 1) | 683 | accuracy **80.8%** | V1 `legal_topic_classification` |
| 2.4 | Kiểm tra nhận định (Đúng/Sai) | 599 | accuracy **81.8%** | V2 `judgment_verification` |
| 2.5 | Phân loại intent (MCQ 4, nhiều đúng) | 1359 | macro-F1 **0.424** | V3 `user_intent_understanding` |

## Ba vấn đề data cần sửa, xếp theo mức thiệt hại

| # | Vấn đề | Nhóm | Thiệt hại đo được |
|---|---|---|---|
| 1 | **Sập về đơn nhãn** — train không có mẫu 3–4 nhãn | V3 | recall 0.338; đúng **0.2%** khi cần 2 nhãn, **0%** khi cần 4 |
| 2 | **Lệch phân bố intent** — `legal_query` thiếu nghiêm trọng | V3 | bỏ sót nhãn này **865 lần** |
| 3 | **Nhầm lẫn lĩnh vực giáp ranh** — không phải do mất cân bằng | V1 | 101 ca phân loại sai |

V2 **không có vấn đề về data** (đã cân bằng 50/50, độ dài khớp benchmark) — lỗi ở đây đến từ bản thân bộ đề, xem mục Task 2.4.

## Task 2.5 — nghiêm trọng nhất

### Vấn đề 1: sập về đơn nhãn

| Số nhãn/câu | Data train | Benchmark | Model dự đoán |
|---|---|---|---|
| 1 | 621 (79.6%) | 397 (29.2%) | 1282 (94.3%) |
| 2 | 159 (20.4%) | 453 (33.3%) | 61 (4.5%) |
| 3 | 0 (0.0%) | 358 (26.3%) | 13 (1.0%) |
| 4 | 0 (0.0%) | 151 (11.1%) | 0 (0.0%) |
| **Trung bình** | **1.20** | **2.19** | **1.06** |

**Data train không có một mẫu 3 hoặc 4 nhãn nào.** Model học theo và chọn 1 nhãn ở 1282/1359 câu (94%). Hệ quả trực tiếp:

| Gold cần | Tỉ lệ model đúng trọn bộ |
|---|---|
| 1 nhãn | 44.3% (176/397) |
| 2 nhãn | 0.2% (1/453) |
| 3 nhãn | 0.3% (1/358) |
| 4 nhãn | 0.0% (0/151) |

**Cần sửa:** bổ sung mẫu 3 nhãn (~26% tổng) và 4 nhãn (~11%), đưa trung bình từ 1.20 lên khoảng 2.2.

### Vấn đề 2: lệch phân bố intent

| Intent | Train | Benchmark | Lệch | Bỏ sót | Thêm thừa | Cần làm |
|---|---|---|---|---|---|---|
| `chitchat` | 8.8% | 5.6% | -3.3 | 108 | 55 | giữ |
| `comparative_analysis` | 17.4% | 11.8% | -5.7 | 208 | 47 | giảm |
| `document_relationship` | 14.9% | 8.4% | -6.5 | 169 | 47 | giảm |
| `document_retrieval` | 11.9% | 16.8% | +4.9 | 379 | 30 | giữ |
| `external_analysis` | 14.7% | 1.1% | -13.6 | 21 | 41 | giảm |
| `general` | 11.5% | 6.6% | -4.9 | 84 | 193 | giữ |
| `legal_query` | 20.9% | 43.1% | +22.2 | 865 | 3 | **tăng mạnh** |
| `stats_summary` | 20.1% | 6.6% | -13.5 | 139 | 19 | giảm |

`legal_query` là nhãn phổ biến nhất benchmark (43.1%) nhưng chỉ chiếm 17.4% data train — model bỏ sót nó **865 lần** và gần như không bao giờ thêm thừa (3 lần). Ngược lại `general` bị thêm thừa **193 lần**.

### Vấn đề 3: gán nhãn thiếu trong data train

**320/780 mẫu (41%)** có dấu hiệu ngôn ngữ của một intent nhưng intent đó không nằm trong nhãn — hầu hết là thiếu `legal_query`. Ví dụ:

- `"Cho mình hỏi tác động kinh tế - xã hội của Điều 6 trong văn bản 40/2025/TT-BYT so với quy định trước đó"` → nhãn `[comparative_analysis, external_analysis]`, thiếu `legal_query` dù có nhắc Điều 6.
- `"So sánh Điều 65 Nghị định 23/2024 với điều khác?"` → nhãn `[comparative_analysis]`, thiếu `legal_query`.

Đối chiếu quy ước benchmark: **98.9% (93/94)** câu benchmark có nhắc Điều/Khoản cụ thể đều được gán `legal_query`.

**Cần sửa:** rà lại toàn bộ nhãn V3 theo quy ước "có nhắc điều/khoản/mục/điểm cụ thể ⇒ thêm `legal_query`". Riêng việc này vừa tăng số nhãn/câu vừa vá đúng nhãn đang thiếu nhất.

## Task 1.2

Sai 131/683 câu. Chia làm hai loại:

| Dạng lỗi | Số ca |
|---|---|
| Bịa đáp án ngoài danh sách (trả lời "G") | 30 |
| Chọn sai lĩnh vực trong A–F | 101 |

### Bịa đáp án — lỗi do cách đóng khung bài toán khi train

MCQ chỉ có A–F nhưng model trả `G`, `G. An toàn thực phẩm`, `G. Quốc phòng - An ninh`, `G. Bình đẳng giới`… Model được train phân loại **mở** trên 27 lĩnh vực nên khi không thấy lĩnh vực nó cho là đúng, nó tạo thêm lựa chọn thay vì chọn phương án gần nhất. Có ca đáp án đúng nằm sẵn trong danh sách mà model vẫn chọn G.

*Đã kiểm chứng:* thêm ràng buộc "bắt buộc chọn trong danh sách, không tạo đáp án mới" vào prompt giảm còn 15 ca và tăng accuracy lên 82.87%. Nhưng gốc rễ nằm ở data: V1 chỉ dạy phân loại mở, chưa từng dạy chọn trong tập cho sẵn.

**Cần sửa:** thêm biến thể MCQ vào V1 — cùng câu hỏi nhưng đưa kèm một tập lựa chọn hữu hạn và yêu cầu chọn trong đó, gồm cả trường hợp không có đáp án hoàn hảo (buộc chọn gần nhất).

### Không có thiên vị vị trí

| Vị trí | Gold | Model đoán | Lệch |
|---|---|---|---|
| A | 17.1% | 14.1% | -3.1 |
| B | 17.7% | 15.7% | -2.0 |
| C | 16.3% | 15.2% | -1.0 |
| D | 16.5% | 17.6% | +1.0 |
| E | 18.9% | 18.7% | -0.1 |
| F | 13.5% | 14.3% | +0.9 |

Lệch tối đa 3.1 điểm — **không có vấn đề thiên vị theo vị trí đáp án**, không cần xáo trộn thứ tự đáp án khi tạo data.

### Thiên vị theo lĩnh vực — không do mất cân bằng train

| Lĩnh vực | Tần suất train | Gold | Model đoán | Lệch |
|---|---|---|---|---|
| Quyền dân sự | 4.1% | 24 | 10 | -14 |
| Đầu tư | 4.0% | 27 | 14 | -13 |
| Thương mại | 5.9% | 30 | 18 | -12 |
| Tiền tệ ngân hàng | 2.1% | 28 | 20 | -8 |
| Tài nguyên - môi trường | 2.1% | 22 | 26 | +4 |
| Kế toán - kiểm toán | 2.1% | 25 | 30 | +5 |
| Dịch vụ pháp lý | 2.1% | 27 | 35 | +8 |
| Xây dựng - đô thị | 2.1% | 25 | 35 | +10 |

Điểm quan trọng: **tần suất trong train không giải thích được thiên vị này**. `Quyền dân sự` chiếm 4.1% train (cao hơn trung bình) nhưng bị đoán thiếu 14 ca; `Xây dựng - đô thị` chỉ 2.1% lại bị đoán thừa 10 ca. Đây là **nhầm lẫn ngữ nghĩa giữa các lĩnh vực giáp ranh**, không phải mất cân bằng số lượng.

Cặp nhầm phổ biến nhất:

- Đầu tư → Xây dựng - đô thị (6 ca)
- Bảo hiểm → Lao động - tiền lương (3 ca)
- Thương mại → Dịch vụ pháp lý (3 ca)
- Quyền dân sự → Bộ máy hành chính (2 ca)
- Đầu tư → Xuất nhập khẩu (2 ca)
- Doanh nghiệp → Đầu tư (2 ca)

**Cần sửa:** thêm mẫu ở đúng các cặp ranh giới này, đặc biệt các ca dễ nhầm (dự án bất động sản ↔ đầu tư ↔ xây dựng; hợp đồng thương mại ↔ dịch vụ pháp lý; quyền nhân thân ↔ thủ tục hành chính). Tăng số lượng đều tay sẽ không giúp — cần mẫu phân định ranh giới.

## Task 2.4 — lỗi không nằm ở data train

Sai 109/599 câu: **83 ca nhận nhầm** (gold=Sai, model nói Đúng) và 26 ca bác nhầm. Model nói "Đúng" ở 58.6% số câu so với tỉ lệ thực 49.1%.

Data V2 đã cân bằng hoàn hảo (390 Đúng / 390 Sai) và độ dài khớp benchmark, nên thiên vị này **không đến từ mất cân bằng nhãn**. Nguyên nhân thật:

**Một phần câu hỏi không thể trả lời từ thông tin được cấp.** Nhận định hỏi về phán quyết của tòa phúc thẩm, nhưng bản án chỉ dừng ở sơ thẩm và việc kháng cáo.

| Nhóm | n | Accuracy |
|---|---|---|
| Nhận định nhắc "phúc thẩm" | 95 | 72.6% |
| Không nhắc | 504 | 83.5% |

Với các ca gold=Sai, trung bình **52% số từ trong `grounding` không xuất hiện trong bản án** — tức căn cứ để phán Sai nằm ngoài đầu vào của model.

Ví dụ điển hình: nhận định nói *"phúc thẩm cho phép phát mãi cả hai thửa đất **1053** và **1911**"*, trong khi bản án chỉ ghi "một số thửa đất" — hai số thửa này **không hề xuất hiện** trong bản án. Không ai, kể cả người đọc, phán được câu này.

**Cần sửa (ở bộ đề, không phải data train):** bổ sung phán quyết cuối cùng vào trường `description`, hoặc loại nhóm câu này khi báo cáo năng lực model.

**Lỗi thật của model** chỉ còn khoảng 47 ca: bản án có đủ dữ kiện, nhận định đảo ngược một chi tiết (tòa "chấp nhận" ↔ "bác", có ↔ không nghĩa vụ bồi thường), nhưng model chỉ kiểm tra tính hợp lý bề mặt. Với nhóm này, thêm mẫu V2 nhấn vào phép đảo ngược tinh vi sẽ có ích.

## Thứ tự ưu tiên

| Ưu tiên | Việc | Nhóm | Kỳ vọng |
|---|---|---|---|
| 1 | Bổ sung mẫu 3–4 nhãn, nâng trung bình lên ~2.2 nhãn/câu | V3 | tăng recall từ 0.338 |
| 2 | Rà lại nhãn theo quy ước "nhắc điều/khoản ⇒ thêm `legal_query`" | V3 | vá 865 ca bỏ sót |
| 3 | Cân lại phân bố intent theo cột "Cần làm" ở bảng trên | V3 | giảm 193 ca thừa `general` |
| 4 | Thêm biến thể MCQ (chọn trong tập cho sẵn) | V1 | hết 30 ca bịa đáp án |
| 5 | Thêm mẫu phân định các cặp lĩnh vực giáp ranh | V1 | giảm 101 ca nhầm |
| 6 | Thêm mẫu nhấn vào phép đảo ngược chi tiết | V2 | giảm ~47 ca nhận nhầm |

**Không cần làm:** xáo trộn vị trí đáp án (không có thiên vị vị trí), cân bằng lại nhãn Đúng/Sai của V2 (đã 50/50), mở rộng độ dài context (chỉ 0.2% câu vượt `max_length=2048`).
