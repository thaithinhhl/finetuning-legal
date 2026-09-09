# Fine-tuning Qwen 7B 4-bit cho dữ liệu pháp lý

Codebase này dùng **QLoRA**: base model được nạp ở NF4 4-bit, còn các adapter LoRA được huấn luyện. Train/dev/test được giữ tách biệt: train dùng cập nhật trọng số, dev chọn checkpoint tốt nhất, test chỉ được đánh giá sau khi train.

## 1. Yêu cầu

- Linux, Python 3.10 hoặc 3.11 và NVIDIA GPU.
- Khuyến nghị VRAM 16 GB trở lên cho `max_length: 2048`, batch size 1. Nếu OOM, giảm `max_length` xuống 1024 trước.
- Không nên chạy cấu hình bitsandbytes này trên macOS/Apple Silicon.

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

Schema `messages` và `instruction` chỉ tính loss trên phần trả lời của assistant; schema `text` tính loss trên toàn văn bản. Sửa đường dẫn, model và siêu tham số trong `configs/qwen2_5_7b_qlora.yaml`.

## 3. Kiểm tra rồi train

```bash
PYTHONPATH=. python validate_data.py --config configs/qwen2_5_7b_qlora.yaml
PYTHONPATH=. accelerate launch train.py --config configs/qwen2_5_7b_qlora.yaml
```

Adapter tốt nhất nằm tại `outputs/qwen2.5-7b-legal-qlora/best_adapter`; metrics train/dev/test nằm trong cùng thư mục. Nếu job bị ngắt, đặt `training.resume_from_checkpoint` thành đường dẫn checkpoint gần nhất.

Test nhanh:

```bash
PYTHONPATH=. python inference.py \
  --adapter outputs/qwen2.5-7b-legal-qlora/best_adapter \
  --prompt "Hãy giải thích điều kiện có hiệu lực của hợp đồng."
```

## 4. Vì sao mặc định 3 epochs?

Ba epochs là baseline hợp lý cho supervised fine-tuning, nhưng không phải luôn tối ưu. Code đánh giá dev sau mỗi epoch và tự nạp checkpoint có `eval_loss` thấp nhất. Theo dõi đường train/dev loss:

- train loss giảm, dev loss tăng: đang overfit; giảm còn 1–2 epochs, giảm learning rate hoặc tăng dữ liệu;
- cả hai còn giảm ở cuối epoch 3: có thể thử thêm epoch;
- dữ liệu ít: nên đánh giá thêm chất lượng câu trả lời thủ công, không chỉ dựa vào loss.

Effective batch size mặc định là `1 × 16 × số GPU`. Nếu đổi `eval_strategy` và `save_strategy` sang `steps`, hãy giữ hai giá trị strategy giống nhau và đặt `eval_steps` bằng `save_steps`.
