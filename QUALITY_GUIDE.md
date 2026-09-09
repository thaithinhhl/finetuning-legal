# Hướng dẫn nâng cao chất lượng fine-tuning mô hình pháp lý

Chất lượng sau fine-tuning phụ thuộc chủ yếu vào dữ liệu và cách đánh giá. GPU mạnh, nhiều epoch hoặc LoRA rank cao không thể bù cho dữ liệu sai, dữ liệu trùng hoặc test leakage.

## 1. Làm sạch và kiểm tra dữ liệu

Trước khi train, cần kiểm tra:

- Căn cứ pháp lý và kết luận trong output có chính xác không.
- Văn bản được viện dẫn có đúng phạm vi, thời điểm và còn hiệu lực không.
- Không có output rỗng, mâu thuẫn, quá ngắn hoặc sai format.
- Không có dữ liệu cá nhân hoặc dữ liệu nhạy cảm chưa được xử lý.
- Train/dev/test không chứa mẫu trùng hoặc gần trùng.
- Các câu hỏi từ cùng một tài liệu, vụ án hoặc tình huống gốc phải nằm trong cùng một split.
- Dữ liệu đủ đa dạng về lĩnh vực, độ khó, độ dài và cách đặt câu hỏi.
- Có mẫu thiếu dữ kiện để model học cách yêu cầu bổ sung thông tin thay vì tự suy đoán.

Không nên đánh giá chất lượng data chỉ bằng số lượng. Nghiên cứu [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) cho thấy dữ liệu instruction được chọn lọc kỹ có thể mang lại hiệu quả cao dù số lượng nhỏ.

## 2. Chuẩn hóa câu trả lời pháp lý

Có thể sử dụng cấu trúc sau cho các câu hỏi cần phân tích:

```text
Kết luận ngắn gọn:
...

Căn cứ pháp lý:
- Điều ... Luật ... năm ...

Phân tích:
...

Lưu ý/ngoại lệ:
...
```

Không bắt buộc mọi output phải dài. Câu hỏi đơn giản nên có câu trả lời ngắn; tình huống phức tạp mới cần phân tích đầy đủ.

System prompt và chat template khi inference phải giống lúc training. Pipeline hiện sử dụng chat template đi cùng tokenizer của Qwen và chỉ tính loss trên câu trả lời của assistant. Xem thêm hướng dẫn [Hugging Face Chat Templates](https://huggingface.co/docs/transformers/chat_templating).

## 3. Kiểm soát độ dài và truncation

Cấu hình mặc định dùng `max_length: 2048`. Trước khi train cần thống kê:

- Phân vị 50%, 90%, 95% và 99% của tổng số token.
- Số token của prompt và answer riêng biệt.
- Tỷ lệ mẫu vượt quá `max_length`.
- Tỷ lệ answer bị cắt một phần hoặc toàn bộ.

Nên chọn `max_length` bao phủ khoảng 90–95% dữ liệu và dành trước một lượng token tối thiểu cho answer, ví dụ 512 token. Không nên để prompt dài chiếm toàn bộ sequence khiến model không còn answer để học.

## 4. Baseline và ma trận thí nghiệm

Cấu hình BF16 LoRA hiện tại là baseline:

```yaml
lora:
  r: 32
  alpha: 64
  dropout: 0.05

training:
  learning_rate: 0.0001
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
```

Effective batch size trên một GPU là `2 × 16 = 32`.

Không thay đổi nhiều tham số cùng lúc. Nên bắt đầu với bốn run:

| Run | LoRA rank | Learning rate | Epochs |
|---|---:|---:|---:|
| Baseline | 32 | `1e-4` | 3 |
| LR thấp | 32 | `5e-5` | 3 |
| Rank cao | 64 | `1e-4` | 3 |
| Rank cao, LR thấp | 64 | `5e-5` | 3 |

Một run ba epoch đã lưu checkpoint theo từng epoch, vì vậy có thể so sánh epoch 1, 2 và 3 mà không cần train lại.

Có thể thử `use_rslora: true` với rank 32 hoặc 64. rsLoRA sử dụng cách scale ổn định hơn khi rank tăng. Xem [PEFT LoRA documentation](https://huggingface.co/docs/peft/main/conceptual_guides/lora).

## 5. Đánh giá chất lượng

Không chọn model chỉ dựa trên `eval_loss`. Cần xây dựng một bộ evaluation khoảng 100–300 câu hỏi đại diện và không dùng các câu này để điều chỉnh data train.

Các tiêu chí chấm:

1. Kết luận có đúng không.
2. Điều luật và văn bản viện dẫn có chính xác không.
3. Văn bản có còn hiệu lực và đúng thời điểm áp dụng không.
4. Lập luận có bám sát dữ kiện không.
5. Model có thêm dữ kiện không được cung cấp không.
6. Model có nhận biết trường hợp thiếu thông tin không.
7. Model có thể hiện sự thận trọng khi tồn tại ngoại lệ không.
8. Câu trả lời có rõ ràng, đúng format và phù hợp độ dài không.

Nên so sánh mù giữa:

- Base Qwen chưa fine-tune.
- Checkpoint epoch 1.
- Checkpoint epoch 2.
- Checkpoint epoch 3.

Tất cả model phải nhận cùng prompt và sử dụng cùng cấu hình decoding, ví dụ `do_sample=False`.

## 6. Kết hợp RAG

Fine-tuning không nên được dùng như cách duy nhất để model ghi nhớ toàn bộ văn bản pháp luật.

- Fine-tuning giúp model học cách trả lời, lập luận, trình bày và xử lý trường hợp thiếu dữ kiện.
- RAG cung cấp điều luật và văn bản hiện hành từ nguồn chính thức.
- Prompt nên yêu cầu model chỉ viện dẫn căn cứ xuất hiện trong context được truy xuất.

Thiết kế này giúp giảm rủi ro model viện dẫn văn bản đã hết hiệu lực hoặc tự tạo điều luật.

## 7. Nhận biết overfitting và underfitting

### Overfitting

- Train loss tiếp tục giảm nhưng dev loss tăng.
- Model lặp nguyên văn câu trả lời trong train.
- Model áp dụng cùng một khuôn trả lời cho nhiều loại câu hỏi.
- Khả năng trả lời câu hỏi tổng quát giảm rõ rệt.

Cách xử lý: giảm còn 1–2 epoch, giảm learning rate, tăng dropout, giảm rank hoặc cải thiện độ đa dạng data.

### Underfitting

- Train loss và dev loss đều còn cao.
- Dev loss vẫn giảm mạnh ở cuối epoch 3.
- Model chưa học được format hoặc thuật ngữ mong muốn.

Cách xử lý: kiểm tra lại data trước, sau đó thử rank 64, tăng epoch có kiểm soát hoặc bổ sung dữ liệu chất lượng cao.

## 8. Checklist trước khi thuê GPU

- [ ] Xác nhận đúng model ID và license.
- [ ] Xác nhận schema của cả ba split.
- [ ] Kiểm tra exact duplicate và near-duplicate.
- [ ] Kiểm tra leakage theo tài liệu/vụ việc gốc.
- [ ] Kiểm tra tính đúng và hiệu lực của căn cứ pháp lý.
- [ ] Thống kê độ dài prompt/answer và tỷ lệ truncation.
- [ ] Chạy thành công `validate_data.py`.
- [ ] Chuẩn bị bộ câu hỏi evaluation độc lập.
- [ ] Chạy thử một tập nhỏ để xác nhận loss giảm và checkpoint lưu được.
- [ ] Sau smoke test mới chạy đầy đủ ba epoch.
- [ ] Chỉ đánh giá test set sau khi đã khóa cấu hình bằng dev set.

## 9. Thứ tự cải thiện code tiếp theo

1. Dành trước token cho answer khi truncate.
2. Bổ sung kiểm tra near-duplicate và leakage theo nguồn tài liệu.
3. Thêm script generation evaluation và rubric pháp lý.
4. Cho phép bật/tắt rsLoRA trong config.
5. Lưu metadata của mỗi run: package, CUDA, GPU, seed, config và commit Git.
