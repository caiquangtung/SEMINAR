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
python3 -m pip install -r requirements.txt
```

### 2) Tiền xử lý dữ liệu

```bash
python3 data_utils.py --max_len 20 --show_stats
```

### 3) Điền TODO trong model.py, sau đó tự kiểm tra

```bash
python3 model.py
```

Kỳ vọng khi điền đúng:

- scaled_dot_product_attention ... PASSED
- SelfAttention ... PASSED
- FeedForwardNetwork ... PASSED
- TransformerEncoderBlock ... PASSED

### 4) Huấn luyện

```bash
python3 train.py
python3 train.py --run_all
python3 train.py --d_model 128 --d_ff 256
```

Ý nghĩa nhanh:

- `python3 train.py`: chạy 1 cấu hình Transformer mặc định.
- `python3 train.py --run_all`: chạy đủ 4 model (3 Transformer + 1 MLP baseline).
- `python3 train.py --d_model 128 --d_ff 256`: chạy 1 cấu hình tùy chỉnh.

### 5) Visualize attention

```bash
python3 visualize.py
python3 visualize.py --model results/model_Transformer_d128_ff256.pt
python3 visualize.py --sentence "this film is absolutely terrible"
```

Lưu ý:

- Nếu không truyền `--model`, chương trình tự chọn 1 model Transformer đầu tiên theo thứ tự tên file.
- `visualize.py` chỉ áp dụng cho Transformer, không áp dụng cho `MLPBaseline_d64`.

## Bộ lệnh đầy đủ để chạy tất cả model và đủ yêu cầu

Chạy theo đúng thứ tự dưới đây:

### A) Cài thư viện

```bash
python3 -m pip install -r requirements.txt
```

### B) Tiền xử lý dữ liệu

```bash
python3 data_utils.py --max_len 20 --show_stats
```

### C) Kiểm tra TODO trong model.py

```bash
python3 model.py
```

### D) Train tất cả model bắt buộc

```bash
python3 train.py --run_all
```

Lệnh này chạy đủ 4 model:

- Transformer_d64_ff128
- Transformer_d128_ff256
- Transformer_d32_ff64
- MLPBaseline_d64

Kết quả lưu trong thư mục `results/`:

- `model_*.pt`
- `learning_curve_*.png`
- `summary.json`

Lưu ý quan trọng:

- `summary.json` luôn ghi theo lần chạy gần nhất.
- Nếu chạy lại cùng dữ liệu + cùng tham số + cùng seed, kết quả có thể giống hệt lần trước.

### E) Visualize theo yêu cầu báo cáo (ít nhất 3 câu)

```bash
python3 visualize.py --model results/model_Transformer_d128_ff256.pt --sentence "this movie is wonderful and inspiring"
python3 visualize.py --model results/model_Transformer_d128_ff256.pt --sentence "the plot is boring and predictable"
python3 visualize.py --model results/model_Transformer_d128_ff256.pt --sentence "this film is not good at all"
```

Mỗi lần chạy sẽ ghi đè `results/attention_heatmap.png`.

### F) Visualize lần lượt tất cả model Transformer đang có

```bash
python3 visualize.py --model results/model_Transformer_d32_ff64.pt --sentence "this movie is not good"
mv results/attention_heatmap.png results/attention_heatmap_Transformer_d32_ff64.png

python3 visualize.py --model results/model_Transformer_d64_ff128.pt --sentence "this movie is not good"
mv results/attention_heatmap.png results/attention_heatmap_Transformer_d64_ff128.png

python3 visualize.py --model results/model_Transformer_d128_ff256.pt --sentence "this movie is not good"
mv results/attention_heatmap.png results/attention_heatmap_Transformer_d128_ff256.png
```

### G) One-shot (chạy nhanh pipeline bắt buộc)

```bash
python3 -m pip install -r requirements.txt && \
python3 data_utils.py --max_len 20 --show_stats && \
python3 model.py && \
python3 train.py --run_all
```

## Ghi chú cho sinh viên

- Chỉ cần điền các phần `# TODO` trong model.py.
- Không đổi tên hàm và tham số.
- Dữ liệu đã có sẵn cột `split`.
- Bộ dữ liệu này là dữ liệu mô phỏng cân bằng 3 lớp để phục vụ học Transformer.
