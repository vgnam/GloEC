# GloEC: Hướng dẫn Training Model

Dự án **GloEC** — mô hình dự đoán chức năng enzyme dựa trên cấu trúc phân cấp (hierarchical-aware global model).

---

## 1. Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|------------|---------|
| **OS** | Windows 10/11, Linux, macOS |
| **Python** | **3.10** (khuyến nghị) |
| **GPU** | NVIDIA GPU với VRAM >= 4GB (hỗ trợ CUDA) |
| **CUDA** | >= 11.8 |
| **Dung lượng ổ đĩa** | ~10GB (bao gồm model ESM-2 650M ~2.5GB) |

### Các package chính (Python 3.10)

```bash
py -3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
py -3.10 -m pip install transformers pandas scikit-learn numpy loguru tqdm torch-summary tensorboardX
```

**Kiểm tra GPU:**
```bash
py -3.10 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## 2. Cấu trúc thư mục quan trọng

```
GloEC/
├── 01.Code/                    # Mã nguồn chính
│   ├── train.py                # Script training chính
│   ├── predict.py              # Script dự đoán
│   ├── config_util.py          # Cấu hình hyperparameters
│   ├── create_esm_hf.py        # Tạo ESM embeddings (Hugging Face)
│   ├── ESM.py                  # Kiến trúc model GloEC
│   ├── model_util.py           # Utilities cho model
│   ├── dataset_util.py         # DataLoader và tiền xử lý
│   └── ...
├── 02.Datasets/
│   └── uniport_2022_5/
│       ├── f_train.csv         # Data train
│       ├── f_eval.csv          # Data validation
│       ├── f_time_test.csv     # Data test
│       ├── f_conti_label_map.json   # Label mapping
│       ├── esm_f_train.pt      # ESM embeddings (sẽ tạo)
│       ├── esm_f_eval.pt       # ESM embeddings (sẽ tạo)
│       └── esm_f_time_test.pt  # ESM embeddings (sẽ tạo)
├── Save_model/                 # Thư mục lưu model checkpoint
├── logs/                       # Thư mục lưu log training
└── fold_data/                  # Thư mục lưu index k-fold
```

---

## 3. Chuẩn bị dữ liệu

### 3.1 Tạo các thư mục cần thiết

```bash
cd D:\GloEC
mkdir Save_model
mkdir logs
mkdir fold_data
```

### 3.2 Tạo ESM Embeddings

Model GloEC không nhận trực tiếp chuỗi amino acid, mà cần **ESM-2 embeddings** (1280 chiều) đã được pre-compute.

**Sử dụng Hugging Face (khuyến nghị):**

```bash
cd 01.Code
py -3.10 create_esm_hf.py
```

Script sẽ:
- Tự động tải model `facebook/esm2_t33_650M_UR50D` về (~2.5GB, lần đầu)
- Tạo 3 file `.pt` trong `02.Datasets/uniport_2022_5/`:
  - `esm_f_train.pt`
  - `esm_f_eval.pt`
  - `esm_f_time_test.pt`

**Lưu ý:**
- Lần đầu chạy có thể mất **15-30 phút** tùy thuộc vào tốc độ mạng và GPU.
- Nếu gặp lỗi **Out of Memory (OOM)**, mở `create_esm_hf.py` và giảm `BATCH_SIZE` xuống `2` hoặc `1`.
- Nếu không có GPU, script vẫn chạy được trên CPU nhưng rất chậm.

---

## 4. Training

### 4.1 Cấu hình Training

Mở file `01.Code/config_util.py` để điều chỉnh các hyperparameters:

| Parameter | Mô tả | Giá trị mặc định |
|-----------|-------|------------------|
| `model_name` | Tên model | `'ESM'` |
| `epoch` | Số epoch train | `160` |
| `batch_size` | Batch size | `512` |
| `learning_rate` | Learning rate | `0.001` |
| `iskfold` | Bật 10-fold cross validation | `True` |
| `kfold_epoch` | Số epoch cho mỗi fold | `40` |
| `use_GCN` | Sử dụng GCN | `True` |
| `gcn_layer` | Số layer GCN | `3` |
| `device` | Thiết bị train | Tự động (`cuda:0` hoặc `cpu`) |

**Các thay đổi phổ biến:**

```python
# Ví dụ: Tăng batch size nếu GPU mạnh
self.batch_size = 512

# Ví dụ: Tắt k-fold để train nhanh
self.iskfold = False
self.epoch = 160

# Ví dụ: Tiếp tục train từ checkpoint cũ
self.is_continue_train = True
self.continue_train_num = '06131214'  # Tên model cũ
```

### 4.2 Chạy Training

```bash
cd D:\GloEC\01.Code
py -3.10 train.py
```

**Quá trình training:**
1. Load ESM embeddings từ `02.Datasets/uniport_2022_5/`
2. Khởi tạo model GloEC
3. Train và validate qua từng epoch
4. Tự động lưu model tốt nhất vào `../Save_model/ESM_<timestamp>.pth`
5. Ghi log chi tiết vào `../logs/<timestamp>.log`

**Kết quả in ra màn hình:**
```
epoch1 -train: loss:0.1234   time:45.2s
-eval: layer1_f1:0.852   layer2_f1:0.781   layer3_f1:0.654   layer4_f1:0.432
lr = [0.001]
```

### 4.3 Chế độ K-Fold Cross Validation

Mặc định `iskfold = True`, model sẽ train theo **10-fold stratified cross-validation**:

- Mỗi fold train ~40 epoch (`kfold_epoch`)
- Model tốt nhất của mỗi fold được lưu: `ESM_<timestamp>_1.pth`, `ESM_<timestamp>_2.pth`, ...
- Có thể dừng sớm sau fold đầu tiên bằng cách đặt `sure_full_kfold = False`

---

## 5. Dự đoán (Inference)

### 5.1 Chuẩn bị

Mở `01.Code/predict.py`, sửa các dòng sau:

```python
model_index = 'ESM_06131214'   # <-- Đổi thành tên model bạn đã train
# Ví dụ: model_index = 'ESM_05101530'
```

### 5.2 Chạy dự đoán

```bash
cd D:\GloEC\01.Code
py -3.10 predict.py
```

**Kết quả:**
- In ra F1, Precision, Recall cho 4 tầng EC number
- File CSV kết quả được lưu tại `02.Datasets/predict_result/`

---

## 6. Troubleshooting

| Lỗi | Nguyên nhân | Cách khắc phục |
|-----|-------------|----------------|
| `ModuleNotFoundError: No module named 'torch'` | PyTorch chưa cài cho Python 3.10 | `py -3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| `CUDA out of memory` | VRAM không đủ | Giảm `batch_size` trong `config_util.py` hoặc `create_esm_hf.py` |
| `FileNotFoundError: ../Save_model/` | Thiếu thư mục | Chạy `mkdir Save_model` |
| `FileNotFoundError: esm_f_train.pt` | Chưa tạo embeddings | Chạy `py -3.10 create_esm_hf.py` |
| `No files found in Save_model` | Chưa train model | Chạy `py -3.10 train.py` trước |
| Model chạy trên CPU dù có GPU | PyTorch CPU-only | Cài lại PyTorch với CUDA: `py -3.10 -m pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `ImportError` từ `sklearn` | Version scikit-learn không tương thích | `py -3.10 -m pip install scikit-learn --upgrade` |

---

## 7. Lệnh tổng hợp (Quick Start)

```bash
# 1. Vào thư mục project
cd D:\GloEC

# 2. Tạo thư mục
mkdir Save_model, logs, fold_data

# 3. Cài package (nếu chưa có)
py -3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
py -3.10 -m pip install transformers pandas scikit-learn numpy loguru tqdm torch-summary tensorboardX

# 4. Tạo ESM embeddings
cd 01.Code
py -3.10 create_esm_hf.py

# 5. Train model
py -3.10 train.py

# 6. Dự đoán (sau khi train xong)
py -3.10 predict.py
```

---

## 8. Liên hệ

Nếu có câu hỏi hoặc gặp vấn đề với code, vui lòng liên hệ:  
**Yiran Huang** — hyr@gxu.edu.cn
