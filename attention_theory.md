# Attention Theory for Sentiment Classification

Tai lieu nay tong hop ly thuyet va cach phan tich attention trong do an Mini Transformer.
Muc tieu la giup ban giai thich duoc mo hinh, khong chi bao cao chi so accuracy.

## 1. Attention la gi?

Trong Transformer, moi token trong cau se tao ra 3 bieu dien:

- Query (Q): token dang di tim thong tin
- Key (K): token cung cap "dia chi" thong tin
- Value (V): token cung cap noi dung thong tin

Self-attention tinh muc do token i can "nhin" token j.
Ket qua la ma tran weights (he so chu y) de tron thong tin toan cau.

## 2. Cong thuc cot loi

Scaled dot-product attention:

- scores = Q @ K^T / sqrt(d_k)
- weights = softmax(scores, dim=-1)
- output = weights @ V

Y nghia:

- Chia cho sqrt(d_k) giup score khong qua lon, softmax on dinh hon.
- softmax theo chieu cuoi dam bao moi hang trong weights co tong xap xi 1.
- output la to hop co trong so cua cac vector V.

## 3. Shape tensor can nho

Gia su:

- batch_size = B
- seq_len = L
- hidden size = d_k

Thi:

- Q, K, V: (B, L, d_k)
- scores: (B, L, L)
- weights: (B, L, L)
- output: (B, L, d_k)

Trong ma tran weights:

- Hang i: token i dang "hoi"
- Cot j: token j duoc token i "nhin"

## 4. Vi sao can attention visualization?

Train metrics cho biet mo hinh manh hay yeu, nhung khong cho biet "vi sao".
Heatmap attention giup:

- Xem token nao chi phoi du doan
- Kiem tra mo hinh co nhan ra phu dinh (not, never) hay khong
- Phan tich truong hop du doan sai

Tom lai:

- Accuracy = danh gia dinh luong
- Attention heatmap = giai thich dinh tinh

## 5. Heatmap la gi va doc nhu the nao

Heatmap la anh cua ma tran attention weights kich thuoc L x L.

Thong thuong:

- Truc X: key tokens (token duoc nhin)
- Truc Y: query tokens (token dang nhin)
- Mau dam/nhat: attention cao/thap (tuy colormap)

Cach doc 1 o (i, j):

- Token o hang i dang dat muc chu y nao do vao token o cot j.

Dau hieu "hop ly":

- Tu cam xuc (great, terrible, boring...) duoc chu y cao
- Tu phu dinh (not, never) duoc chu y trong cac cau phu dinh
- Chu y khong qua ngau nhien vao stop words

## 6. Gioi han cua attention interpretation

Can luu y:

1. Attention khong phai bang chung tuyet doi ve feature importance.
2. Heatmap chi cho thay mo hinh "nhin" o dau, khong chung minh nhan qua truc tiep.
3. Mot cau co the co nhieu cach chu y hop ly khac nhau.
4. Can ket hop voi test/val metrics va error analysis de ket luan.

## 7. Lien he voi bai toan sentiment

Trong bai toan 3 lop (negative, neutral, positive):

- Positive: ky vong chu y vao tu tich cuc (excellent, amazing, wonderful...)
- Negative: ky vong chu y vao tu tieu cuc (awful, terrible, boring...)
- Neutral: ky vong chu y phan tan hon, it token cam xuc manh

Cau phu dinh la case quan trong:

- "not good" co y nghia am tinh, khong phai tich cuc
- Neu mo hinh chu y qua nhieu vao "good" ma bo qua "not", de du doan sai

## 8. Quy trinh phan tich attention trong bao cao

Nen chon it nhat 3 nhom cau:

1. Cau du doan dung
2. Cau du doan sai
3. Cau co phu dinh

Moi cau nen bao cao:

- Cau goc
- Nhan dung
- Nhan du doan
- Anh heatmap
- 2-3 nhan xet ngan

Mau nhan xet:

- "Model tap trung vao token X va Y, phu hop voi nhan negative."
- "Attention bi phan tan, khong co token noi bat, dan den du doan sai."
- "Token not khong duoc chu y du, model bi nham nghia cua good."

## 9. Lien ket voi kien truc trong do an

Trong project nay:

- Attention duoc tinh trong ham `scaled_dot_product_attention`
- Weights duoc luu de visualize
- `visualize.py` tao heatmap tu attention weights cua model Transformer

MLP baseline khong co self-attention nen khong co heatmap attention tuong ung.

## 10. Chon model tot nhat: dung ca dinh luong va dinh tinh

De chon model hop ly:

1. Xep hang theo test_accuracy
2. Neu gan nhau, xem val_accuracy va do on dinh cua learning curve
3. Dung heatmap de kiem tra model co chu y hop ly hay khong

Neu hai model co test_accuracy bang nhau:

- Uu tien model nhe hon de de trien khai
- Hoac uu tien model co attention de giai thich hon

## 11. Loi thuong gap khi visualize

1. Khong chi dinh model, script tu chon model Transformer dau tien

- Dan den ban nghi dang xem model A nhung thuc te la model B

2. Ghi de attention_heatmap.png

- Moi lan chay co the mat anh cu
- Nen doi ten anh sau moi lan chay

3. Dung cau qua dai

- Sau cat/pad max_len, mot so token quan trong co the bi cat bo

4. Danh dong attention voi giai thich tuyet doi

- Can ket hop error analysis

## 12. Checklist viet phan Attention Analysis trong report

- Co it nhat 3 heatmap
- Co ca truong hop dung va sai
- Co it nhat 1 cau phu dinh
- Moi heatmap co 2-3 nhan xet ro rang
- Co tong ket chung: model hoc duoc gi, thieu gi

## 13. Ket luan ngan

Visualize attention la cong cu giup ban tra loi:

- Mo hinh dang chu y vao dau?
- Vi sao no du doan dung/sai?

Trong do an nay, phan attention la diem then chot de the hien ban hieu ban chat Transformer, vuot ra ngoai viec chi bao cao mot con so accuracy.
