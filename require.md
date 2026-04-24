# Xây dựng Transformer cho bài toán phân loại cảm xúc văn bản

Môn học: **Seminar chuyên đề**  
Chủ đề: **Cài đặt Transformer Encoder đơn giản (PyTorch) cho phân loại cảm xúc văn bản**

## 1) Tổng quan

Trong đồ án này, sinh viên tự cài đặt một **Transformer Encoder** đơn giản bằng **Python/PyTorch** để giải quyết bài toán **phân loại cảm xúc văn bản**.

- **Mục tiêu chính**: hiểu cơ chế **Self-Attention** hoạt động như thế nào và có khả năng **phân tích/giải thích** kết quả mô hình (không đặt nặng việc đạt accuracy cao nhất).

## 2) Bài toán

- **Đầu vào**: Câu tiếng Anh ngắn (\< 20 từ sau khi cắt/padding theo `max_len`).
- **Đầu ra**: Nhãn cảm xúc **Positive / Negative / Neutral** (3 lớp).
- **Dữ liệu**: Tập giả lập do giảng viên cung cấp:
  - 600 mẫu, đã chia sẵn **train/val/test = 420/90/90**
  - Cân bằng **200 mẫu mỗi nhãn**

## 3) Phạm vi cài đặt (được dùng thư viện gì?)

Quy tắc chung: **được phép** dùng thư viện (PyTorch, sklearn, matplotlib, …) cho mọi thành phần trong Transformer **ngoại trừ**:

- **Self-Attention**: *bắt buộc tự cài đặt* bằng các phép tính ma trận cơ bản.
- **FFN (Feed-Forward Network)**: *bắt buộc tự cài đặt* (chỉ dùng `nn.Linear` + activation).

Những phần được phép dùng thư viện:

- Add & LayerNorm (residual): dùng `nn.LayerNorm`
- PositionalEncoding: dùng code mẫu giảng viên cấp
- Embedding/Optimizer/Loss: `nn.Embedding`, `Adam`, `CrossEntropyLoss`
- Tokenization, vẽ đồ thị: tuỳ thư viện

## 4) Tài nguyên được cung cấp (skeleton)

Giảng viên cung cấp sẵn dữ liệu và phần lớn code hỗ trợ; sinh viên chỉ cần điền các phần `# TODO`.

Cấu trúc thư mục sau khi giải nén:

```text
MSSV_HoTen_DoAn/
├── data/
│   ├── sentiment_raw.csv          # 600 câu có nhãn, có sẵn cột split
│   └── processed/                 # tạo bởi data_utils.py
│       ├── train.pt
│       ├── val.pt
│       ├── test.pt
│       ├── vocab.json
│       └── meta.json
├── data_utils.py                  # tiền xử lý dữ liệu (có sẵn)
├── model.py                       # kiến trúc Transformer (SV điền TODO)
├── train.py                       # huấn luyện + thực nghiệm (có sẵn)
├── visualize.py                   # heatmap attention (có sẵn)
├── requirements.txt
└── README.md
```

## 5) Hướng dẫn chạy (theo thứ tự)

### Bước 1 — Cài thư viện

```bash
pip install -r requirements.txt

# Nếu máy bạn không có lệnh `python` (macOS hay gặp), dùng:
python3 -m pip install -r requirements.txt
```

### Bước 2 — Tiền xử lý dữ liệu (chỉ cần chạy 1 lần)

```bash
python data_utils.py --max_len 20 --show_stats

# Hoặc:
python3 data_utils.py --max_len 20 --show_stats
```

Kết quả tạo ra trong `data/processed/`: `train.pt`, `val.pt`, `test.pt`, `vocab.json`, `meta.json`.

### Bước 3 — Kiểm tra `model.py` (sau khi điền TODO)

```bash
python model.py

# Hoặc:
python3 model.py
```

Kỳ vọng khi điền đúng:

- `TEST: scaled_dot_product_attention ... PASSED`
- `TEST: SelfAttention ... PASSED`
- `TEST: FeedForwardNetwork ... PASSED`
- `TEST: TransformerEncoderBlock ... PASSED`

### Bước 4 — Huấn luyện

```bash
python train.py                  # cấu hình mặc định (d_model=64, d_ff=128)
python train.py --run_all         # chạy >= 3 cấu hình + baseline MLP
python train.py --d_model 128     # thay đổi siêu tham số

# Hoặc:
python3 train.py
python3 train.py --run_all
python3 train.py --d_model 128 --d_ff 256
```

`train.py` sẽ lưu model tốt nhất vào `results/model_*.pt` và các kết quả (learning curve, summary) trong `results/`.

### Bước 5 — Visualize attention

```bash
python visualize.py
python visualize.py --model results/model_Transformer_d128_ff256.pt
python visualize.py --sentence "this film is absolutely terrible"

# Hoặc:
python3 visualize.py
python3 visualize.py --model results/model_Transformer_d128_ff256.pt
python3 visualize.py --sentence "this film is absolutely terrible"
```

Lưu ý: cần chạy `train.py` (Bước 4) trước khi visualize.

## 6) Mô tả dữ liệu `sentiment_raw.csv`

Mỗi dòng là một câu đã gán nhãn, ví dụ các cột:

- `id`: 1–600
- `split`: `train` / `val` / `test`
- `text`: câu tiếng Anh
- `label`: 0/1/2
- `label_name`: `negative` / `neutral` / `positive`
- `num_tokens`: số từ trong câu

Ghi chú:

- Cột `split` đã điền sẵn; `data_utils.py` dùng cột này để tách tập.

## 7) Yêu cầu kỹ thuật chi tiết

### 7.1 Kiến trúc mô hình

Mô hình gồm đúng **1** khối Transformer Encoder theo thứ tự:

1. **Embedding**: `nn.Embedding` (token id → vector kích thước `d_model`)
2. **Positional Encoding**: cộng thông tin vị trí (dùng code mẫu)
3. **1 × TransformerEncoderBlock**, gồm:
   - **Self-Attention**: tính Q, K, V → attention score → context vector (**tự cài đặt**)
   - **Add & LayerNorm** (residual) (**dùng `nn.LayerNorm`**)
   - **Feed-Forward Network**: `Linear(d_model, d_ff) -> ReLU -> Linear(d_ff, d_model)` (**tự cài đặt**)
   - **Add & LayerNorm** lần 2 (**dùng `nn.LayerNorm`**)
4. **Classifier head**: mean pooling → `Linear(d_model, num_classes)` (dùng code mẫu)

Lý do chỉ dùng 1 block: dataset nhỏ (600 mẫu), nhiều block dễ overfitting; mục tiêu là hiểu sâu 1 block.

### 7.2 Hiểu về tensor và batch (rất quan trọng)

Trong PyTorch, mọi tensor thường có thêm chiều batch ở đầu.

Ví dụ (batch_size=32, seq_len=10, d_model=64):

- `x.shape = (32, 10, 64)`
- `Q.shape = (32, 10, 64)`
- `K.shape = (32, 10, 64)`
- `V.shape = (32, 10, 64)`
- `scores.shape = (32, 10, 10)` với `scores = Q @ K.transpose(-2, -1) / sqrt(d_k)`
- `weights.shape = (32, 10, 10)` sau `softmax(dim=-1)` (mỗi hàng tổng ~ 1.0)
- `output.shape = (32, 10, 64)`

Gợi ý debug: in `.shape` sau mỗi bước; sai shape thì dừng sửa trước khi đi tiếp.

### 7.3 Yêu cầu về Self-Attention (bắt buộc)

Hàm `scaled_dot_product_attention(Q, K, V)` phải:

- Nhận `Q, K, V` shape `(batch, seq_len, d_k)`
- Tính `scores = Q @ K^T / sqrt(d_k)` → shape `(batch, seq_len, seq_len)`
- Áp dụng `softmax` trên **chiều cuối** (`dim=-1`) để ra `weights`
- Tính `output = weights @ V`
- Trả về **(output, weights)** (để visualize)

Kiểm tra tối thiểu:

- Với `Q=K=V=torch.randn(2, 10, 32)`:
  - `output.shape == (2, 10, 32)`
  - `weights.shape == (2, 10, 10)`
  - `weights.sum(dim=-1) ≈ ones`

### 7.4 Siêu tham số gợi ý

- `d_model = 64`
- `d_ff = 128` (≈ 2 × `d_model`)
- `max_len = 20`
- `batch_size = 32`
- `lr = 1e-3` (Adam)
- `num_epochs = 30–50`

### 7.5 Skeleton code đã có sẵn (SV không cần tự viết)

- `data_utils.py`: đọc CSV, build vocab, tokenize, padding, lưu `data/processed/`
- `model.py`: framework (PositionalEncoding, ClassifierHead, TransformerClassifier) — SV chỉ điền `# TODO`
- `train.py`: train/val loop, learning curve, baseline MLP
- `visualize.py`: vẽ heatmap attention cho câu bất kỳ

Lưu ý: **không đổi tên hàm và tham số** để tương thích bộ test.

## 8) Lộ trình thực hiện (gợi ý)

- **Tuần 1**: hiểu luồng dữ liệu + cài `scaled_dot_product_attention`, pass unit test
- **Tuần 2**: cài FFN + ghép EncoderBlock + train end-to-end, theo dõi learning curve
- **Tuần 3**: chạy nhiều cấu hình, visualize attention, viết báo cáo, đóng gói

Checkpoint tham khảo:

- Tuần 2: mô hình train được, val loss giảm; val acc > 70% (tham khảo).

## 9) Yêu cầu thực nghiệm

### 9.1 Baseline bắt buộc

Phải chạy và báo cáo kết quả baseline **MLP** có sẵn trong `train.py`.

```bash
python train.py --run_all
```

### 9.2 Thực nghiệm tối thiểu (>= 3 cấu hình Transformer)

Báo cáo ít nhất 3 cấu hình:

- Cấu hình 1 (mặc định): `d_model=64`, `d_ff=128`, `batch_size=32`
- Cấu hình 2: `d_model=128`, `d_ff=256`
- Cấu hình 3: `d_model=32`, `d_ff=64`

Với mỗi cấu hình (kể cả baseline MLP), báo cáo:

- `train_accuracy`, `val_accuracy`, `test_accuracy`
- `final_train_loss` (loss cuối cùng)

Có thể đề xuất cấu hình thứ 4 tuỳ chọn.

### 9.3 Visualization attention weights

Dùng `visualize.py` để tạo heatmap (ma trận `seq_len × seq_len`).

Yêu cầu tối thiểu:

- Chọn > 3 câu từ tập test
  - Câu 1: dự đoán đúng → attention tập trung từ nào? có hợp lý không?
  - Câu 2: dự đoán sai → attention bị phân tán hay tập trung sai chỗ?
  - Câu 3: có phủ định (“not”, “never”, “don’t”) → mô hình có nhận ra phủ định không?
- Với mỗi heatmap: viết **2–3 câu nhận xét**

## 10) Yêu cầu báo cáo (PDF)

Nộp báo cáo PDF dài **6–10 trang** (không tính phụ lục), gồm các mục:

1. **Mô tả kiến trúc** (1–2 trang)
   - Sơ đồ kiến trúc tổng thể (tự vẽ, không lấy ảnh internet)
   - Mô tả thành phần tự cài đặt: công thức toán + giải thích
   - Vì sao cần Add & LayerNorm sau mỗi sublayer
2. **Kết quả thực nghiệm** (2–3 trang)
   - Bảng so sánh tất cả cấu hình + baseline MLP
   - Learning curve của cấu hình tốt nhất
   - Phân tích: cấu hình nào tốt nhất? có overfitting không? bắt đầu từ epoch nào?
3. **Phân tích attention** (1–2 trang)
   - 3 heatmap + nhận xét
   - Nhận xét tổng quát: mô hình chú ý loại từ nào? có hợp lý không?
4. **Error analysis** (1 trang)
   - 5–10 câu dự đoán sai (nhãn đúng vs nhãn dự đoán)
   - Phân nhóm lỗi (phủ định, mơ hồ, từ lạ/OOV…)
   - Đề xuất 1–2 hướng cải thiện cụ thể
5. **Kết luận** (~0.5 trang)
   - Tóm tắt đã làm được gì
   - Bài học quan trọng nhất khi tự cài Self-Attention

## 11) Tiêu chí chấm điểm (tham khảo)

Tổng: **10 điểm**

- Self-Attention (Q,K,V): đúng shape, scaling, softmax, gradient flow — **3.5**
- FFN & ghép EncoderBlock: FFN 2 lớp tự viết; Add & Norm dùng thư viện; forward chạy được — **2.0**
- Thực nghiệm & so sánh: >= 4 cấu hình (gồm baseline), learning curve, nhận xét — **2.0**
- Visualization: heatmap > 3 câu, đúng chiều, nhận xét có ý nghĩa — **1.0**
- Báo cáo & phân tích lỗi: rõ ràng, có insight, error analysis — **1.5**

## 12) Quy định nộp bài

### 12.1 Cấu trúc nộp

Nộp 1 file `.zip` tên `MSSV_HoTen_DoAn.zip`, gồm:

```text
MSSV_HoTen_Seminar_Codes/
├── data/
│   └── sentiment_raw.csv        # giữ nguyên
├── model.py
├── train.py
├── visualize.py
├── data_utils.py
├── requirements.txt
├── README.md                    # hướng dẫn chạy (<= 1 trang)
└── MSSV_HoTen_Seminar_Report.pdf
```

Lưu ý: **không nộp** `data/processed/` (các file `.pt` nặng, giảng viên sẽ chạy lại `data_utils.py` để tạo).

### 12.2 Yêu cầu về code

- Chạy được theo thứ tự:
  - `pip install -r requirements.txt`
  - `python data_utils.py`
  - `python train.py`
- Không dùng đường dẫn tuyệt đối (chỉ dùng đường dẫn tương đối)
- Kết quả tái tạo được: set random seed (ví dụ `torch.manual_seed(42)`)
- Có comment giải thích cho mỗi hàm và bước tính toán quan trọng

### 12.3 Quy định làm bài

- Được tham khảo tài liệu/sách/bài báo gốc
- Được thảo luận ý tưởng, nhưng code/báo cáo phải độc lập
- Ghi rõ nguồn tham khảo trong báo cáo
- Nghiêm cấm: sao chép code; dùng AI sinh code phần tự cài đặt
- Khi vấn đáp: cần giải thích được từng dòng code trong `scaled_dot_product_attention` (không giải thích được sẽ bị trừ điểm nặng)

## 13) Câu hỏi vấn đáp tham khảo

- Q/K/V có shape như thế nào? chiều nào là batch, chiều nào là seq_len?
- Vì sao phải chia cho \( \sqrt{d_k} \)? Nếu không chia thì gradient có thể ra sao khi \(d_k\) lớn?
- Attention weights phải thoả điều kiện gì? kiểm tra thế nào trong code?
- Nếu hai từ giống nhau hoàn toàn, attention weights giữa chúng trông như thế nào?
- Residual connection (Add) có tác dụng gì? bỏ đi sẽ thế nào?
- Vì sao dùng LayerNorm thay vì BatchNorm?
- FFN trong block có bao nhiêu tham số? tính thế nào?

## 14) Điểm cộng (tuỳ chọn)

- (+5) Multi-Head Attention (`num_heads > 1`): tách Q/K/V thành nhiều head, tính song song, ghép lại
- (+5) Padding mask: bỏ qua vị trí `[PAD]` khi tính attention
- (+3) Dropout sau attention weights và sau FFN; phân tích ảnh hưởng overfitting

> Điểm cộng chỉ áp dụng khi điểm bắt buộc đạt từ 70/100 trở lên.

---

## Phần triển khai (đã hoàn thành)

### 1) Vị trí code đã triển khai

- Thư mục code: `Codes_Data/`
- Các phần bắt buộc tự cài đặt nằm ở `Codes_Data/model.py`:
  - `scaled_dot_product_attention(Q, K, V)`
  - `FeedForwardNetwork` (2 lớp Linear + ReLU)

### 2) Chạy end-to-end và output

Chạy trong thư mục `Codes_Data/` theo thứ tự:

```bash
python3 -m pip install -r requirements.txt
python3 data_utils.py --max_len 20 --show_stats
python3 model.py
python3 train.py --num_epochs 20
python3 visualize.py --sentence "this film is absolutely terrible"
```

Output sinh ra:

- `Codes_Data/data/processed/`: `train.pt`, `val.pt`, `test.pt`, `vocab.json`, `meta.json`
- `Codes_Data/results/`: `model_*.pt`, `learning_curve_*.png`, `summary.json`, `attention_heatmap.png`

