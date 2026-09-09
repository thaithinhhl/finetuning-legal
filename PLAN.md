# Kế hoạch thí nghiệm

## Pha 1 — xác nhận dữ liệu

1. Chốt schema và ý nghĩa của từng cột với người cung cấp dữ liệu.
2. Chạy `validate_data.py`: số mẫu, độ dài token, câu trả lời bị truncate và exact overlap giữa các split.
3. Kiểm tra thủ công một mẫu ngẫu nhiên ở mỗi split; đặc biệt là căn cứ pháp lý, nhãn và dữ liệu nhạy cảm.
4. Giữ nguyên test set cho đến lần đánh giá cuối; mọi lựa chọn siêu tham số chỉ dựa trên dev.

## Pha 2 — baseline QLoRA

1. Base: `Qwen/Qwen2.5-7B-Instruct` (thay đúng checkpoint nếu model đầu vào khác).
2. Quantization: NF4 4-bit, double quantization, bf16; LoRA rank 16 trên các linear projection của Qwen.
3. Train 3 epochs, effective batch size 16/GPU, learning rate `2e-4`, cosine schedule, warmup 3%.
4. Log thường xuyên, evaluate/save sau mỗi epoch và chọn checkpoint có dev loss thấp nhất.

## Pha 3 — đánh giá

1. Báo cáo train loss, dev loss và test loss/perplexity của checkpoint tốt nhất.
2. Lập một bộ câu hỏi pháp lý đại diện để chấm thủ công: đúng căn cứ, đúng kết luận, đầy đủ, không bịa điều luật và mức độ an toàn.
3. So sánh base model với adapter trên đúng cùng prompt và decoding (`do_sample=False`).
4. Chỉ sau khi khóa cấu hình mới chạy test và ghi kết quả cuối.

## Pha 4 — vòng thí nghiệm tiếp theo (nếu cần)

- Overfit: thử 1–2 epochs, giảm LR xuống `1e-4`, hoặc tăng LoRA dropout.
- Underfit: thử rank 32, thêm epoch, hoặc tăng dữ liệu chất lượng cao.
- OOM: giảm `max_length` 2048 → 1024; sau đó mới cân nhắc giảm batch/LoRA rank.
- Câu trả lời bị cắt nhiều: phân tích phân vị độ dài rồi chọn `max_length`, không tăng mù vì attention tốn bộ nhớ theo độ dài.

Mỗi run cần lưu config, seed, mã nguồn/commit, phiên bản package, GPU và metrics để có thể tái lập.
