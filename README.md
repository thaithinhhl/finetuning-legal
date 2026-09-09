# Fine-tuning Qwen 7B bằng BF16 LoRA cho dữ liệu pháp lý

Codebase này nạp base model ở **BF16**, đóng băng base weights và huấn luyện adapter **LoRA**. Không có lượng tử 4-bit trong training hoặc inference. Train/dev/test được giữ tách biệt: train dùng cập nhật adapter, dev chọn checkpoint tốt nhất, test chỉ được đánh giá sau khi train.

## 1. Yêu cầu

- Linux, Python 3.10 hoặc 3.11 và NVIDIA GPU.
- Khuyến nghị NVIDIA H100/H200; cấu hình mặc định được thiết kế cho một H200 141 GB.
- BF16 phải được GPU hỗ trợ. Pipeline không chạy training trên macOS/Apple Silicon hoặc CPU.
- Cần đủ dung lượng đĩa để cache base model BF16 và lưu checkpoint.

```bash
python -m venv .venv
source .venv/bin/activate
# Cài PyTorch đúng với CUDA trên máy trước, sau đó:
pip install -r requirements.txt
```

## 2. Đặt dữ liệu

Mặc định đặt ba file tại `data/train.jsonl`, `data/dev.jsonl`, `data/test.jsonl`. JSON, JSONL, CSV và Parquet đều được hỗ trợ, nhưng ba split phải cùng loại file. Chọn một trong ba schema:

**Chat (`messages`)**

```json
{"messages":[{"role":"user","content":"Thời hiệu khởi kiện là gì?"},{"role":"assistant","content":"..."}]}
```

**Instruction**

```json
{"instruction":"Trả lời câu hỏi", "input":"Thời hiệu khởi kiện là gì?", "output":"..."}
```

**Văn bản thuần (`text`)**

```json
{"text":"Nội dung dùng để tiếp tục pretrain..."}
```

Schema `messages` và `instruction` chỉ tính loss trên phần trả lời của assistant; schema `text` tính loss trên toàn văn bản. Sửa đường dẫn, model và siêu tham số trong `configs/qwen2_5_7b_bf16_lora.yaml`.

## 3. Kiểm tra rồi train

```bash
PYTHONPATH=. python validate_data.py --config configs/qwen2_5_7b_bf16_lora.yaml
PYTHONPATH=. accelerate launch train.py --config configs/qwen2_5_7b_bf16_lora.yaml
```

Model được tự tải từ Hugging Face khi chạy lần đầu; có thể đổi `model.name_or_path` thành đường dẫn local. Adapter tốt nhất nằm tại `outputs/qwen2.5-7b-legal-bf16-lora/best_adapter`; metrics train/dev/test nằm trong cùng thư mục. Nếu job bị ngắt, đặt `training.resume_from_checkpoint` thành đường dẫn checkpoint gần nhất.

Test nhanh:

```bash
PYTHONPATH=. python inference.py \
  --adapter outputs/qwen2.5-7b-legal-bf16-lora/best_adapter \
  --prompt "Hãy giải thích điều kiện có hiệu lực của hợp đồng."
```

## 4. Vì sao mặc định 3 epochs?

Ba epochs là baseline hợp lý cho supervised fine-tuning, nhưng không phải luôn tối ưu. Code đánh giá dev sau mỗi epoch và tự nạp checkpoint có `eval_loss` thấp nhất. Theo dõi đường train/dev loss:

- train loss giảm, dev loss tăng: đang overfit; giảm còn 1–2 epochs, giảm learning rate hoặc tăng dữ liệu;
- cả hai còn giảm ở cuối epoch 3: có thể thử thêm epoch;
- dữ liệu ít: nên đánh giá thêm chất lượng câu trả lời thủ công, không chỉ dựa vào loss.

Effective batch size mặc định là `2 × 16 × số GPU`, tức 32 trên một H200. Nếu OOM với dữ liệu dài, giảm `per_device_train_batch_size` từ 2 xuống 1 hoặc giảm `max_length` từ 2048 xuống 1024. Nếu đổi `eval_strategy` và `save_strategy` sang `steps`, hãy giữ hai strategy giống nhau và đặt `eval_steps` bằng `save_steps`.

## 5. Attention trên H200

Mặc định `attn_implementation: sdpa`, không cần cài package ngoài và PyTorch sẽ chọn kernel phù hợp. Nếu muốn dùng package Flash Attention 2 riêng:

```bash
pip install flash-attn --no-build-isolation
```

Sau đó đổi `model.attn_implementation` thành `flash_attention_2`. Không đổi cấu hình này nếu `flash-attn` chưa được cài thành công.
