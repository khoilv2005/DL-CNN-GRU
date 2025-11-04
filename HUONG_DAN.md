# HƯỚNG DẪN SỬ DỤNG - MÔ HÌNH CNN-GRU CHO PHÁT HIỆN XÂM NHẬP IoT

## 📋 MÔ TẢ DỰ ÁN

Dự án này triển khai mô hình Deep Learning CNN-GRU (Convolutional Neural Network - Gated Recurrent Unit) để phát hiện các cuộc tấn công mạng trong hệ thống IoT. Mô hình được xây dựng dựa trên kiến trúc DeepFed, kết hợp sức mạnh của CNN trong việc trích xuất đặc trưng và GRU trong việc học các mẫu tuần tự.

## 🎯 TÍNH NĂNG CHÍNH

- ✅ Phân loại nhị phân: Benign (lành tính) vs Attack (tấn công)
- ✅ Kiến trúc CNN-GRU kết hợp
- ✅ Xử lý dữ liệu tự động (missing values, infinite values, normalization)
- ✅ Visualizations đầy đủ (phân bố dữ liệu, training history, confusion matrix)
- ✅ Lưu trữ model và kết quả chi tiết
- ✅ Callbacks nâng cao (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)

## 📦 YÊU CẦU HỆ THỐNG

### Phần mềm cần thiết:
- Python 3.8 trở lên
- pip (Python package manager)

### Phần cứng khuyến nghị:
- RAM: 8GB trở lên
- GPU: NVIDIA GPU với CUDA (tùy chọn, để tăng tốc training)
- Disk: 5GB trống (cho dataset và model)

## 🚀 CÀI ĐẶT

### Bước 1: Clone hoặc tải dự án về
```bash
# Nếu dùng Git
git clone <repository-url>
cd DeepL

# Hoặc tải file ZIP và giải nén
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)
```powershell
# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
.\venv\Scripts\Activate.ps1

# Nếu gặp lỗi permission, chạy lệnh này trước:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Bước 3: Cài đặt các thư viện
```powershell
# Cài đặt tất cả dependencies
pip install -r requirements.txt

# Hoặc cài đặt từng package riêng lẻ
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow joblib
```

### Bước 4: Kiểm tra cài đặt
```powershell
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import pandas as pd; print('Pandas version:', pd.__version__)"
```

## 📁 CẤU TRÚC THỨ MỤC

```
DeepL/
│
├── DL.py                      # File chính để chạy training
├── requirements.txt           # Danh sách thư viện cần thiết
├── HUONG_DAN.md              # File hướng dẫn này
│
├── IoT_Dataset_2023/         # Thư mục chứa dataset (cần chuẩn bị)
│   ├── file1.csv
│   ├── file2.csv
│   └── ...
│
└── (Các file output sau khi chạy)
    ├── final_cnn_gru_model.h5    # Model đã train
    ├── best_model.h5             # Model tốt nhất trong quá trình train
    ├── scaler.pkl                # StandardScaler để normalize dữ liệu
    ├── label_encoder.pkl         # LabelEncoder cho labels
    ├── results_summary.txt       # Kết quả đánh giá chi tiết
    ├── label_distribution.png    # Biểu đồ phân bố nhãn
    ├── training_history.png      # Quá trình training
    └── confusion_matrix.png      # Ma trận nhầm lẫn
```

## 📊 CHUẨN BỊ DỮ LIỆU

### Yêu cầu Dataset:
1. **Định dạng**: Các file CSV
2. **Vị trí**: Đặt tất cả file CSV trong thư mục `IoT_Dataset_2023`
3. **Cấu trúc**: 
   - Các cột đầu: features (các đặc trưng của traffic mạng)
   - Cột cuối: label (nhãn phân loại, ví dụ: "Benign", "DDoS", "Mirai", v.v.)

### Tải Dataset:
- Dataset IoT 2023 có thể tải từ các nguồn như Kaggle, UCI Machine Learning Repository
- Hoặc sử dụng dataset tùy chỉnh của bạn (đảm bảo định dạng phù hợp)

### Ví dụ cấu trúc file CSV:
```
feature1,feature2,feature3,...,label
0.123,45.67,89.01,...,Benign
1.234,56.78,90.12,...,Attack
...
```

## ▶️ CHẠY CHƯƠNG TRÌNH

### Chạy toàn bộ pipeline:
```powershell
python DL.py
```

### Các bước chương trình sẽ thực hiện:

1. **STEP 1**: Load và phân tích dataset
   - Tìm và đọc tất cả file CSV trong thư mục
   - Merge thành một dataset duy nhất
   - Hiển thị thông tin tổng quan

2. **STEP 2**: Phân tích và thống kê dữ liệu
   - Thống kê phân bố nhãn gốc
   - Chuyển đổi thành binary labels (Benign vs Attack)
   - Tạo biểu đồ phân bố

3. **STEP 3**: Tiền xử lý dữ liệu
   - Xử lý missing values
   - Xử lý infinite values
   - Loại bỏ constant columns
   - Encode labels

4. **STEP 4**: Chia dữ liệu
   - Training: 70%
   - Validation: 10%
   - Test: 20%

5. **STEP 5**: Chuẩn hóa dữ liệu
   - Sử dụng StandardScaler
   - Fit trên training set, transform tất cả các set

6. **STEP 6**: Xây dựng mô hình CNN-GRU
   - CNN Module: 3 Conv blocks
   - GRU Module: 2 GRU layers
   - MLP Module: 2 Dense layers
   - Output: Softmax

7. **STEP 7**: Huấn luyện mô hình
   - Epochs: 50 (có thể dừng sớm với EarlyStopping)
   - Batch size: 128
   - Optimizer: Adam

8. **STEP 8**: Visualize quá trình training
   - Loss curves
   - Accuracy curves
   - Precision & Recall curves

9. **STEP 9**: Đánh giá trên test set
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - Classification Report

10. **STEP 10**: Lưu kết quả
    - Lưu model, scaler, label encoder
    - Lưu báo cáo chi tiết

## 🔧 TÙY CHỈNH THAM SỐ

### Thay đổi đường dẫn dataset:
```python
# Trong file DL.py, dòng 24
DATA_PATH = './IoT_Dataset_2023'  # Thay đổi đường dẫn của bạn
```

### Điều chỉnh hyperparameters:
```python
# Trong file DL.py, dòng 447-448
EPOCHS = 50          # Số epochs
BATCH_SIZE = 128     # Kích thước batch

# Trong file DL.py, dòng 439
learning_rate=0.001  # Learning rate của Adam optimizer
```

### Thay đổi tỷ lệ chia data:
```python
# Trong file DL.py, dòng 163-164
TEST_SIZE = 0.2      # 20% cho test
VAL_SIZE = 0.125     # ~10% cho validation
```

### Điều chỉnh kiến trúc mô hình:
```python
# Trong file DL.py, hàm build_cnn_gru_model (dòng 211-305)
# Thay đổi số filters, units, dropout rates, v.v.
```

## 📈 ĐÁNH GIÁ KẾT QUẢ

### Các metrics được tính:
- **Accuracy**: Tỷ lệ dự đoán đúng tổng thể
- **Precision**: Tỷ lệ dự đoán Attack đúng trong tất cả dự đoán Attack
- **Recall**: Tỷ lệ phát hiện được Attack trong tất cả Attack thực tế
- **F1-Score**: Trung bình điều hòa của Precision và Recall

### Confusion Matrix:
```
                Predicted
                Benign  Attack
Actual Benign     TN      FP
       Attack     FN      TP
```

- **TN (True Negative)**: Dự đoán đúng Benign
- **FP (False Positive)**: Dự đoán nhầm Attack (thực tế là Benign)
- **FN (False Negative)**: Dự đoán nhầm Benign (thực tế là Attack)
- **TP (True Positive)**: Dự đoán đúng Attack

## 🔮 SỬ DỤNG MODEL ĐÃ TRAIN

### Load model và predict:
```python
import numpy as np
import joblib
from tensorflow import keras

# Load model và scaler
model = keras.models.load_model('final_cnn_gru_model.h5')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Chuẩn bị dữ liệu mới (X_new phải có cùng số features)
X_new = np.array([[...]])  # Dữ liệu mới

# Normalize
X_new_scaled = scaler.transform(X_new)

# Predict
predictions = model.predict(X_new_scaled)
predicted_classes = np.argmax(predictions, axis=1)

# Decode labels
predicted_labels = label_encoder.inverse_transform(predicted_classes)

print(f"Prediction: {predicted_labels}")
print(f"Probability: {predictions}")
```

## ⚠️ XỬ LÝ LỖI THƯỜNG GẶP

### Lỗi: "No module named 'tensorflow'"
**Giải pháp**: 
```powershell
pip install tensorflow
```

### Lỗi: "Cannot find dataset"
**Giải pháp**: 
- Kiểm tra đường dẫn `DATA_PATH` trong file `DL.py`
- Đảm bảo thư mục chứa file CSV tồn tại
- Kiểm tra định dạng file (phải là .csv)

### Lỗi: "Out of memory"
**Giải pháp**:
- Giảm `BATCH_SIZE` (ví dụ: 64 hoặc 32)
- Giảm số samples trong dataset
- Đóng các ứng dụng khác đang chạy

### Lỗi: "KeyError" hoặc "ValueError" khi load data
**Giải pháp**:
- Kiểm tra cấu trúc file CSV
- Đảm bảo có cột label ở cuối
- Kiểm tra encoding của file (khuyên dùng UTF-8)

### GPU không được sử dụng
**Giải pháp**:
```python
# Kiểm tra GPU
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Nếu cần cài CUDA và cuDNN cho TensorFlow GPU
# Xem hướng dẫn tại: https://www.tensorflow.org/install/gpu
```

## 📚 TÀI LIỆU THAM KHẢO

### Kiến trúc mô hình:
- DeepFed Paper: Federated Learning với CNN-GRU
- CNN (Convolutional Neural Network): Trích xuất đặc trưng không gian
- GRU (Gated Recurrent Unit): Học các mẫu tuần tự

### Thư viện sử dụng:
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## 🤝 HỖ TRỢ

### Khi gặp vấn đề:
1. Kiểm tra lại phần "Xử lý lỗi thường gặp"
2. Đọc kỹ error message trong terminal
3. Kiểm tra log trong quá trình chạy
4. Xem lại file `results_summary.txt` để biết thông tin chi tiết

## 📝 GHI CHÚ

- Quá trình training có thể mất từ vài phút đến vài giờ tùy thuộc vào:
  - Kích thước dataset
  - Cấu hình phần cứng
  - Số epochs
  - Batch size

- Model tốt nhất sẽ được lưu tự động trong file `best_model.h5` nhờ callback ModelCheckpoint

- Nếu training bị gián đoạn, bạn có thể load lại `best_model.h5` và tiếp tục từ đó

## 🎓 TIPS VÀ TRICKS

### Tăng tốc training:
- Sử dụng GPU nếu có
- Tăng BATCH_SIZE (nếu RAM/VRAM đủ)
- Giảm số epochs nếu model đã converge

### Cải thiện accuracy:
- Thử các learning rates khác nhau (0.01, 0.001, 0.0001)
- Điều chỉnh kiến trúc mô hình (thêm/bớt layers, thay đổi units)
- Thử các techniques khác: Data augmentation, Ensemble learning

### Xử lý imbalanced data:
- Sử dụng class weights
- Oversampling class thiểu số (SMOTE)
- Undersampling class đa số

## ✅ CHECKLIST TRƯỚC KHI CHẠY

- [ ] Đã cài đặt Python 3.8+
- [ ] Đã cài đặt tất cả dependencies trong `requirements.txt`
- [ ] Đã chuẩn bị dataset trong thư mục `IoT_Dataset_2023`
- [ ] Đã kiểm tra đường dẫn `DATA_PATH` trong code
- [ ] Đã kích hoạt virtual environment (nếu dùng)
- [ ] Đủ dung lượng disk (ít nhất 5GB)
- [ ] Đủ RAM (khuyến nghị 8GB+)

## 🎉 CHÚC MỪNG!

Bạn đã sẵn sàng để chạy mô hình CNN-GRU phát hiện xâm nhập IoT!

Chúc bạn training thành công! 🚀
