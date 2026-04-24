# Mini Transformer cho Phân loại Cảm xúc Văn bản

Đây là bộ khung đồ án theo đúng cấu trúc yêu cầu trong đề.

## Cấu trúc thư mục

```text
MSSV_HoTen_DoAn/
├── data/
│   └── sentiment_raw.csv
├── data_utils.py
├── model.py
├── train.py
├── visualize.py
├── requirements.txt
└── README.md
```

## Chạy theo thứ tự

### 1) Cài thư viện

```bash
pip install -r requirements.txt

# Nếu máy bạn không có lệnh `python` (macOS hay gặp), dùng:
python3 -m pip install -r requirements.txt
```

### 2) Tiền xử lý dữ liệu

```bash
python data_utils.py --max_len 20 --show_stats

# Hoặc:
python3 data_utils.py --max_len 20 --show_stats
```

### 3) Điền TODO trong `model.py`, sau đó tự kiểm tra

```bash
python model.py

# Hoặc:
python3 model.py
```

Kỳ vọng khi điền đúng:

- scaled_dot_product_attention ... PASSED
- SelfAttention ... PASSED
- FeedForwardNetwork ... PASSED
- TransformerEncoderBlock ... PASSED

### 4) Huấn luyện

```bash
python train.py
python train.py --run_all
python train.py --d_model 128 --d_ff 256

# Hoặc:
python3 train.py
python3 train.py --run_all
python3 train.py --d_model 128 --d_ff 256
```

### 5) Visualize attention

```bash
python visualize.py
python visualize.py --model results/model_Transformer_d128_ff256.pt
python visualize.py --sentence "this film is absolutely terrible"

# Hoặc:
python3 visualize.py
python3 visualize.py --model results/model_Transformer_d128_ff256.pt
python3 visualize.py --sentence "this film is absolutely terrible"
```

Lưu ý:

- Nếu không truyền `--model`, chương trình sẽ tự chọn 1 file `model_Transformer*.pt` đầu tiên theo thứ tự tên file.
- `visualize.py` chỉ hỗ trợ Transformer, không dùng cho `MLPBaseline_d64`.

## Ghi chú cho sinh viên

- Chỉ cần điền các phần `# TODO` trong `model.py`.
- Không đổi tên hàm và tham số.
- Dữ liệu đã có sẵn cột `split`.
- Bộ dữ liệu này là dữ liệu mô phỏng cân bằng 3 lớp để phục vụ học Transformer.

## Trạng thái hoàn thành (check nhanh)

- `model.py`: đã hoàn thiện `scaled_dot_product_attention` và `FeedForwardNetwork` theo đúng unit test trong file.
- Pipeline đã chạy thành công theo thứ tự: `data_utils.py` → `model.py` → `train.py` → `visualize.py`.
- Kết quả sinh ra:
  - `data/processed/`: `train.pt`, `val.pt`, `test.pt`, `vocab.json`, `meta.json`
  - `results/`: `model_*.pt`, `learning_curve_*.png`, `summary.json`, `attention_heatmap.png`

## Bộ lệnh đầy đủ để chạy tất cả model và đủ yêu cầu

Chạy các lệnh dưới đây theo đúng thứ tự để đáp ứng toàn bộ yêu cầu thực nghiệm.

### A) Cài thư viện

```bash
python3 -m pip install -r requirements.txt
```

### B) Tiền xử lý dữ liệu

```bash
python3 data_utils.py --max_len 20 --show_stats
```

### C) Kiểm tra các TODO trong model.py

```bash
python3 model.py
```

Kỳ vọng phải thấy 4 dòng PASSED cho:

- scaled_dot_product_attention
- SelfAttention
- FeedForwardNetwork
- TransformerEncoderBlock

### D) Train tất cả model bắt buộc (3 Transformer + 1 MLP baseline)

```bash
python3 train.py --run_all
```

Lệnh này tự chạy đủ:

- Transformer_d64_ff128
- Transformer_d128_ff256
- Transformer_d32_ff64
- MLPBaseline_d64

Kết quả lưu tại `results/` gồm model, learning curve và `summary.json` để so sánh.

Lưu ý quan trọng:

- `summary.json` luôn ghi theo lần chạy gần nhất.
- Nếu bạn chạy `python3 train.py` (không `--run_all`) thì `summary.json` chỉ có 1 model.
- Muốn có bảng đủ 4 model trong 1 file, hãy chạy `python3 train.py --run_all`.

### E) (Tuỳ chọn) Train từng cấu hình riêng nếu muốn chạy lại độc lập

```bash
python3 train.py --d_model 64 --d_ff 128
python3 train.py --d_model 128 --d_ff 256
python3 train.py --d_model 32 --d_ff 64
```

### F) Visualize attention theo yêu cầu báo cáo (>= 3 câu)

Ví dụ 3 câu đại diện (đúng/sai/phủ định):

```bash
python3 visualize.py --model results/model_Transformer_d128_ff256.pt --sentence "this movie is wonderful and inspiring"
python3 visualize.py --model results/model_Transformer_d128_ff256.pt --sentence "the plot is boring and predictable"
python3 visualize.py --model results/model_Transformer_d128_ff256.pt --sentence "this film is not good at all"
```

Mỗi lần chạy sẽ tạo/ghi đè `results/attention_heatmap.png`.
Nếu muốn lưu riêng từng ảnh, đổi tên file sau mỗi lần chạy.

### F2) Visualize lần lượt tất cả model Transformer đang có

```bash
python3 visualize.py --model results/model_Transformer_d32_ff64.pt --sentence "this movie is not good"
mv results/attention_heatmap.png results/attention_heatmap_Transformer_d32_ff64.png

python3 visualize.py --model results/model_Transformer_d64_ff128.pt --sentence "this movie is not good"
mv results/attention_heatmap.png results/attention_heatmap_Transformer_d64_ff128.png

python3 visualize.py --model results/model_Transformer_d128_ff256.pt --sentence "this movie is not good"
mv results/attention_heatmap.png results/attention_heatmap_Transformer_d128_ff256.png
```

Ghi chú:

- Không có heatmap cho `MLPBaseline_d64` vì model MLP không có attention weights.

### G) File cần đưa vào báo cáo

- `results/summary.json`: bảng so sánh các model
- `results/learning_curve_*.png`: learning curve cho từng model
- `results/attention_heatmap.png` (hoặc các bản đã đổi tên): ảnh attention để nhận xét

### H) One-shot (chạy nhanh toàn bộ pipeline bắt buộc)

```bash
python3 -m pip install -r requirements.txt && \
python3 data_utils.py --max_len 20 --show_stats && \
python3 model.py && \
python3 train.py --run_all
```
