# TÀI LIỆU GIẢI THÍCH CHI TIẾT CODE CNN-GRU

## 📌 TỔNG QUAN DỰ ÁN

### Mục đích
Xây dựng mô hình Deep Learning để phát hiện các cuộc tấn công mạng (Intrusion Detection) trong hệ thống IoT (Internet of Things) sử dụng kiến trúc CNN-GRU hybrid.

### Kiến trúc Model
- **CNN (Convolutional Neural Network)**: Trích xuất đặc trưng không gian (spatial features)
- **GRU (Gated Recurrent Unit)**: Học các mẫu tuần tự (sequential patterns)
- **MLP (Multi-Layer Perceptron)**: Kết hợp features và phân loại

### Loại bài toán
**Binary Classification**: Phân loại nhị phân
- Class 0: **Benign** (Traffic bình thường, lành tính)
- Class 1: **Attack** (Traffic tấn công)

---

## 📚 CẤU TRÚC CODE CHI TIẾT

### PHẦN 1: IMPORT THƯ VIỆN

```python
import pandas as pd              # Xử lý dữ liệu dạng bảng
import numpy as np               # Tính toán số học, xử lý mảng
import matplotlib.pyplot as plt  # Vẽ biểu đồ
import seaborn as sns           # Vẽ biểu đồ đẹp hơn
from sklearn.model_selection import train_test_split  # Chia train/val/test
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Chuẩn hóa dữ liệu
from sklearn.metrics import *    # Các metrics đánh giá
import tensorflow as tf          # Framework Deep Learning
from tensorflow import keras     # High-level API của TensorFlow
```

**Giải thích**:
- **pandas**: Làm việc với file CSV, xử lý DataFrame
- **numpy**: Tính toán ma trận, mảng, các phép toán số học
- **matplotlib & seaborn**: Tạo các biểu đồ trực quan
- **sklearn**: Công cụ Machine Learning (preprocessing, metrics)
- **tensorflow/keras**: Xây dựng và train Deep Learning model

---

### PHẦN 2: THIẾT LẬP BAN ĐẦU

```python
warnings.filterwarnings('ignore')  # Tắt các cảnh báo không cần thiết

# Set random seed để kết quả reproducible (có thể tái tạo)
np.random.seed(42)
tf.random.set_seed(42)

# Thiết lập style cho biểu đồ
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
```

**Tại sao cần random seed?**
- Đảm bảo kết quả giống nhau mỗi lần chạy
- Quan trọng cho nghiên cứu khoa học, debug
- Số 42 là convention (từ cuốn "The Hitchhiker's Guide to the Galaxy")

---

## 🔄 QUY TRÌNH XỬ LÝ DỮ LIỆU (DATA PIPELINE)

### STEP 1: LOAD VÀ PHÂN TÍCH DATASET

#### 1.1. Tìm tất cả file CSV

```python
DATA_PATH = './IoT_Dataset_2023'
csv_files = []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))
```

**Giải thích**:
- `os.walk()`: Duyệt qua tất cả thư mục con
- Tìm tất cả file có đuôi `.csv`
- Lưu đường dẫn đầy đủ vào list `csv_files`

**Output**: Tìm thấy 63 file CSV

#### 1.2. Load từng file CSV

```python
dfs = []
for file in csv_files:
    try:
        df_temp = pd.read_csv(file)
        dfs.append(df_temp)
        print(f"✓ Loaded: {os.path.basename(file):50s} - {len(df_temp):>10,} samples")
    except Exception as e:
        print(f"✗ Error loading {file}: {e}")
```

**Giải thích**:
- Đọc từng file CSV thành DataFrame
- `try-except`: Bắt lỗi nếu file bị corrupt hoặc format sai
- Hiển thị tên file và số lượng samples

**Ví dụ output**:
```
✓ Loaded: Merged01.csv - 712,311 samples
✓ Loaded: Merged02.csv - 748,585 samples
...
```

#### 1.3. Merge tất cả DataFrame

```python
df = pd.concat(dfs, ignore_index=True)
```

**Giải thích**:
- `pd.concat()`: Gộp tất cả DataFrame theo chiều dọc (vertical)
- `ignore_index=True`: Tạo lại index từ 0 đến n-1
- Kết quả: 1 DataFrame lớn chứa toàn bộ dữ liệu (~45-50 triệu samples)

**Thông tin dataset**:
```
→ Tổng số mẫu: 45,000,000+ samples
→ Số features: 80-100 features
→ Kích thước: (45000000, 85) - example
```

---

### STEP 2: PHÂN TÍCH VÀ THỐNG KÊ DATASET

#### 2.1. Xác định cột Label

```python
label_col = df.columns[-1]  # Cột cuối cùng là label
```

**Giải thích**:
- Dataset IoT thường có label ở cột cuối
- Ví dụ: 'label', 'attack_type', 'category'

#### 2.2. Thống kê nhãn gốc

```python
label_counts = df[label_col].value_counts()
```

**Output mẫu**:
```
PHÂN BỐ NHÃN GỐC:
----------------------------------------------------------------------------------------------------
Tên nhãn                                          Số lượng         Tỷ lệ (%)
----------------------------------------------------------------------------------------------------
Benign                                              20,000,000          44.44%
DDoS                                                10,000,000          22.22%
Mirai                                                8,000,000          17.78%
DoS                                                  5,000,000          11.11%
Recon                                                2,000,000           4.44%
```

#### 2.3. Chuyển đổi thành Binary Labels

```python
def map_to_binary(label):
    label_lower = str(label).lower()
    if 'benign' in label_lower or 'normal' in label_lower:
        return 'Benign'  # Nhãn 0
    else:
        return 'Attack'  # Nhãn 1 (gộp tất cả attack types)

df['binary_label'] = df[label_col].apply(map_to_binary)
```

**Giải thích**:
- Gộp tất cả loại attack thành 1 class "Attack"
- Đơn giản hóa bài toán từ multi-class → binary classification
- `.apply()`: Áp dụng hàm cho từng dòng trong DataFrame

**Output**:
```
PHÂN BỐ SAU KHI GỘP:
----------------------------------------------------------------------------------------------------
Nhãn               Số lượng         Tỷ lệ (%)
----------------------------------------------------------------------------------------------------
Attack              25,000,000          55.56%
Benign              20,000,000          44.44%

→ Tỉ lệ mất cân bằng (Imbalance Ratio): 1.25:1
```

#### 2.4. Visualization - Tạo biểu đồ

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Pie chart - Biểu đồ tròn
axes[0].pie(binary_counts.values, labels=binary_counts.index, 
            autopct='%1.2f%%', colors=['#2ecc71', '#e74c3c'])

# Bar chart - Biểu đồ cột
axes[1].bar(binary_counts.index, binary_counts.values, 
            color=['#2ecc71', '#e74c3c'])
```

**Giải thích**:
- `subplots(1, 2)`: Tạo 2 biểu đồ cạnh nhau (1 hàng, 2 cột)
- Pie chart: Hiển thị tỷ lệ phần trăm
- Bar chart: So sánh số lượng trực quan
- Màu xanh (#2ecc71): Benign
- Màu đỏ (#e74c3c): Attack

**Output**: File `label_distribution.png` được lưu

---

### STEP 3: TIỀN XỬ LÝ DỮ LIỆU

#### 3.1. Tách Features và Labels

```python
X = df.drop([label_col, 'binary_label'], axis=1)  # Features
y = df['binary_label']                             # Labels
```

**Giải thích**:
- `X`: Ma trận features (input cho model)
- `y`: Vector labels (output/target)
- `axis=1`: Xóa theo cột (axis=0 là xóa theo hàng)

**Shape**:
```
X: (45000000, 83) - 45 triệu samples, 83 features
y: (45000000,)    - 45 triệu labels
```

#### 3.2. Xử lý Missing Values (Giá trị thiếu)

```python
print(f"→ Missing values: {X.isnull().sum().sum()}")
if X.isnull().sum().sum() > 0:
    X = X.fillna(0)  # Thay thế bằng 0
```

**Giải thích**:
- `.isnull()`: Kiểm tra từng cell có NULL không
- `.sum().sum()`: Tổng số cell NULL trong toàn bộ DataFrame
- `.fillna(0)`: Thay thế NULL bằng 0

**Tại sao thay bằng 0?**
- Trong network traffic, NULL thường nghĩa là "không có traffic"
- 0 là giá trị an toàn, không làm sai lệch thống kê
- Alternative: Có thể dùng median, mean tùy từng feature

#### 3.3. Xử lý Infinite Values (Giá trị vô cực)

```python
print(f"→ Infinite values: {np.isinf(X.values).sum()}")
if np.isinf(X.values).sum() > 0:
    X = X.replace([np.inf, -np.inf], 0)
```

**Giải thích**:
- `np.inf`: Vô cực dương (+∞)
- `-np.inf`: Vô cực âm (-∞)
- Xảy ra khi chia cho 0: a/0 = ∞

**Ví dụ**:
```python
# Feature: packets_per_second
# packets=1000, time=0 → packets/time = 1000/0 = inf
```

#### 3.4. Chuyển tất cả về Numeric

```python
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(0)
```

**Giải thích**:
- Một số cột có thể có kiểu string, mixed types
- `pd.to_numeric()`: Chuyển sang số
- `errors='coerce'`: Nếu không convert được → NaN
- Sau đó fill NaN bằng 0

#### 3.5. Loại bỏ Constant Columns

```python
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    X = X.drop(constant_cols, axis=1)
```

**Giải thích**:
- Constant column: Cột có tất cả giá trị giống nhau
- Ví dụ: [0, 0, 0, ..., 0] hoặc [5, 5, 5, ..., 5]
- Không mang thông tin → Loại bỏ để giảm chiều dữ liệu

**Tại sao loại bỏ?**
- Không giúp model học được gì
- Tốn memory và computation
- Có thể gây lỗi trong một số algorithms

#### 3.6. Encode Labels

```python
le = LabelEncoder()
y_encoded = le.fit_transform(y)
```

**Giải thích**:
- Chuyển từ text → số
- 'Benign' → 0
- 'Attack' → 1

**Label mapping**:
```python
{'Attack': 0, 'Benign': 1}  # hoặc ngược lại
```

---

### STEP 4: CHIA DỮ LIỆU

#### 4.1. Train / Validation / Test Split

```python
TEST_SIZE = 0.2     # 20% cho test
VAL_SIZE = 0.125    # ~10% cho validation

# Chia train+val và test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y_encoded, 
    test_size=TEST_SIZE,      # 20%
    random_state=42, 
    stratify=y_encoded        # Giữ tỷ lệ class
)

# Chia train và validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=VAL_SIZE,       # 10% of total
    random_state=42,
    stratify=y_train_val
)
```

**Giải thích stratify**:
- Giữ nguyên tỷ lệ giữa các class trong mỗi tập
- Ví dụ: Nếu dataset có 60% Attack, 40% Benign
  - Train set cũng sẽ có ~60% Attack, ~40% Benign
  - Test set cũng sẽ có ~60% Attack, ~40% Benign

**Tỷ lệ cuối cùng**:
```
Training:   70% (~31,500,000 samples)
Validation: 10% (~4,500,000 samples)
Test:       20% (~9,000,000 samples)
```

**Tại sao chia như vậy?**
- **Train set**: Lớn nhất, để model học
- **Validation set**: Tune hyperparameters, early stopping
- **Test set**: Đánh giá cuối cùng (model chưa từng thấy)

---

### STEP 5: CHUẨN HÓA DỮ LIỆU (Normalization)

#### 5.1. StandardScaler

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

**Công thức StandardScaler**:
```
z = (x - μ) / σ

Trong đó:
- x: Giá trị gốc
- μ (mu): Mean (trung bình)
- σ (sigma): Standard deviation (độ lệch chuẩn)
- z: Giá trị đã chuẩn hóa
```

**Ví dụ cụ thể**:
```python
# Feature: packet_size
# Values: [64, 128, 256, 512, 1024]
# Mean (μ) = 396.8
# Std (σ) = 369.4

# Chuẩn hóa:
# 64   → (64 - 396.8) / 369.4 = -0.90
# 128  → (128 - 396.8) / 369.4 = -0.73
# 256  → (256 - 396.8) / 369.4 = -0.38
# 512  → (512 - 396.8) / 369.4 = 0.31
# 1024 → (1024 - 396.8) / 369.4 = 1.70
```

**Tại sao cần chuẩn hóa?**
1. **Các features có scale khác nhau**:
   - packet_size: 0-65535
   - packet_count: 0-1000000
   - duration: 0-3600
2. **Neural Network hoạt động tốt hơn** với dữ liệu chuẩn hóa
3. **Tránh features có giá trị lớn** dominate quá trình training
4. **Giúp gradient descent hội tụ nhanh hơn**

**Lưu ý quan trọng**:
- `fit_transform()` chỉ dùng cho **train set**
- `transform()` cho val và test (dùng μ và σ từ train set)
- **KHÔNG fit trên val/test** để tránh data leakage!

---

## 🧠 XÂY DỰNG MÔ HÌNH CNN-GRU

### STEP 6: KIẾN TRÚC MODEL

#### 6.1. Tổng quan kiến trúc

```
Input (83 features)
    |
    ├─────────────────┬─────────────────┐
    |                 |                 |
CNN Module        GRU Module      (Parallel)
    |                 |
Conv Block 1      GRU Layer 1
Conv Block 2      GRU Layer 2
Conv Block 3          |
    |                 |
  Flatten             |
    |                 |
    └─────────────────┴─────────────────┐
                      |
                 Concatenate
                      |
                 MLP Module
                      |
                Dense Layer 1 (256 units)
                Dense Layer 2 (128 units)
                      |
                Output Layer (2 classes)
                      |
                  Softmax
```

#### 6.2. Input Layer

```python
input_layer = layers.Input(shape=input_shape, name='input')
# input_shape = (83,) - 83 features
```

**Giải thích**:
- Nhận input vector 83 chiều
- Shape: (batch_size, 83)

#### 6.3. CNN Module - Trích xuất đặc trưng không gian

**Conv Block 1**:
```python
x_cnn = layers.Reshape((input_shape[0], 1), name='reshape_cnn')(input_layer)
# (batch, 83) → (batch, 83, 1)

x_cnn = layers.Conv1D(filters=64, kernel_size=3, padding='same', 
                      activation='relu', name='conv1')(x_cnn)
# Output: (batch, 83, 64)

x_cnn = layers.BatchNormalization(name='bn1')(x_cnn)
# Chuẩn hóa output của conv layer

x_cnn = layers.MaxPooling1D(pool_size=2, name='pool1')(x_cnn)
# Output: (batch, 41, 64) - giảm một nửa
```

**Giải thích từng layer**:

1. **Reshape**: 
   - Chuyển từ vector (83,) → matrix (83, 1)
   - Cần thiết cho Conv1D

2. **Conv1D**:
   - `filters=64`: Tạo 64 feature maps
   - `kernel_size=3`: Cửa sổ trượt size 3
   - `padding='same'`: Giữ nguyên kích thước
   - `activation='relu'`: Hàm kích hoạt ReLU

3. **BatchNormalization**:
   - Chuẩn hóa output
   - Tăng tốc training, tránh overfitting

4. **MaxPooling1D**:
   - Giảm kích thước xuống 1/2
   - Giữ lại giá trị max trong mỗi window
   - Giảm computation, tăng receptive field

**Conv Block 2 & 3**: Tương tự nhưng tăng số filters (128, 256)

**Flatten**:
```python
cnn_output = layers.Flatten(name='flatten_cnn')(x_cnn)
# Output: (batch, features_dim)
```

#### 6.4. GRU Module - Học mẫu tuần tự

```python
x_gru = layers.Reshape((input_shape[0], 1), name='reshape_gru')(input_layer)
# (batch, 83, 1)

x_gru = layers.GRU(units=128, return_sequences=True, name='gru1')(x_gru)
# Output: (batch, 83, 128)

x_gru = layers.GRU(units=64, return_sequences=False, name='gru2')(x_gru)
# Output: (batch, 64)
```

**Giải thích GRU**:
- **GRU (Gated Recurrent Unit)**: Biến thể của LSTM, đơn giản hơn
- **return_sequences=True**: Trả về output cho mọi timestep
- **return_sequences=False**: Chỉ trả về output cuối cùng

**Tại sao dùng GRU?**
- Học được temporal dependencies (phụ thuộc thời gian)
- Trong network traffic, có thể có pattern theo thời gian
- Ví dụ: Tốc độ tăng dần, burst traffic, v.v.

#### 6.5. Concatenate - Kết hợp CNN và GRU

```python
concatenated = layers.Concatenate(name='concatenate')([cnn_output, gru_output])
```

**Giải thích**:
- Ghép output của CNN và GRU theo chiều features
- CNN: Spatial features
- GRU: Temporal features
- Combined: Cả hai loại features

#### 6.6. MLP Module - Phân loại

```python
# Dense Layer 1
x = layers.Dense(256, activation='relu', name='dense1')(concatenated)
x = layers.BatchNormalization(name='bn_mlp1')(x)
x = layers.Dropout(0.5, name='dropout1')(x)
# Dropout 50%: Randomly tắt 50% neurons

# Dense Layer 2
x = layers.Dense(128, activation='relu', name='dense2')(x)
x = layers.BatchNormalization(name='bn_mlp2')(x)
x = layers.Dropout(0.3, name='dropout2')(x)
# Dropout 30%
```

**Giải thích Dropout**:
- Randomly "tắt" một số neurons trong training
- Tránh overfitting (model học quá kỹ training data)
- 0.5 = tắt 50%, 0.3 = tắt 30%
- Chỉ hoạt động trong training, không dùng trong inference

#### 6.7. Output Layer

```python
output = layers.Dense(num_classes, activation='softmax', name='output')(x)
# num_classes = 2 (Benign, Attack)
```

**Softmax activation**:
```
Softmax(x_i) = exp(x_i) / Σ exp(x_j)

Output: [0.3, 0.7]
        ↓     ↓
     Benign Attack
```

**Giải thích**:
- Chuyển logits thành xác suất
- Tổng các xác suất = 1
- Ví dụ: [0.3, 0.7] → 30% Benign, 70% Attack

#### 6.8. Compile Model

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)
```

**Giải thích các thành phần**:

1. **Optimizer: Adam**
   - Adaptive Moment Estimation
   - Kết hợp momentum và adaptive learning rate
   - learning_rate=0.001: Bước nhảy trong gradient descent

2. **Loss: Sparse Categorical Crossentropy**
   ```
   Loss = -Σ y_true * log(y_pred)
   ```
   - Sparse: Labels là integers (0, 1) thay vì one-hot
   - Đo sự khác biệt giữa dự đoán và ground truth

3. **Metrics**:
   - **Accuracy**: Tỷ lệ dự đoán đúng
   - **Precision**: Trong các dự đoán Attack, bao nhiêu % đúng
   - **Recall**: Trong các Attack thực tế, bao nhiêu % được phát hiện

---

### STEP 7: HUẤN LUYỆN MÔ HÌNH

#### 7.1. Hyperparameters

```python
EPOCHS = 50          # Số lần duyệt qua toàn bộ dataset
BATCH_SIZE = 128     # Số samples trong 1 batch
```

**Giải thích**:
- **1 Epoch**: Model xem qua tất cả training samples 1 lần
- **Batch**: Chia nhỏ data thành các batch để train
- **Batch size 128**: Mỗi lần update weights, dùng 128 samples

**Tại sao không train toàn bộ dataset cùng lúc?**
- Dataset quá lớn (45M samples) không fit vào RAM/GPU
- Mini-batch gradient descent nhanh hơn và ổn định hơn

#### 7.2. Callbacks

**EarlyStopping**:
```python
EarlyStopping(
    monitor='val_loss',        # Theo dõi validation loss
    patience=10,               # Chờ 10 epochs
    restore_best_weights=True  # Khôi phục weights tốt nhất
)
```

**Giải thích**:
- Nếu val_loss không giảm sau 10 epochs → Dừng training
- Tránh lãng phí thời gian khi model đã converge
- Restore best weights: Dùng model tốt nhất, không phải model cuối

**ReduceLROnPlateau**:
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,                # Giảm learning rate xuống 50%
    patience=5,
    min_lr=1e-7
)
```

**Giải thích**:
- Nếu val_loss không giảm sau 5 epochs → Giảm learning rate
- lr_new = lr_old * 0.5
- Giúp model tìm được minimum tốt hơn

**ModelCheckpoint**:
```python
ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',    # Theo dõi val accuracy
    save_best_only=True        # Chỉ lưu model tốt nhất
)
```

**Giải thích**:
- Tự động lưu model tốt nhất trong quá trình training
- Không cần phải train lại nếu muốn dùng best model

#### 7.3. Training Process

```python
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)
```

**Quá trình training mỗi epoch**:
```
1. Forward pass: Tính output từ input
2. Compute loss: So sánh output vs ground truth
3. Backward pass: Tính gradient
4. Update weights: weights -= lr * gradient
5. Evaluate trên validation set
6. Check callbacks (early stopping, reduce LR, etc.)
```

**Output mẫu**:
```
Epoch 1/50
246094/246094 [==============================] - 450s 2ms/step
loss: 0.2534 - accuracy: 0.9245 - precision: 0.9156 - recall: 0.9345
val_loss: 0.2145 - val_accuracy: 0.9367 - val_precision: 0.9287 - val_recall: 0.9445
Epoch 00001: val_accuracy improved from -inf to 0.93674, saving model to best_model.h5

Epoch 2/50
...
```

---

### STEP 8: VISUALIZE TRAINING HISTORY

#### 8.1. Plot Loss & Metrics

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Loss curve
axes[0, 0].plot(history.history['loss'], label='Train Loss')
axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
```

**Giải thích các biểu đồ**:

1. **Loss Curve**:
   - Train loss giảm: Model đang học
   - Val loss giảm: Model generalize tốt
   - Val loss tăng: Có thể bị overfitting

2. **Accuracy Curve**:
   - Thể hiện % dự đoán đúng qua mỗi epoch
   - Train acc thường cao hơn val acc

3. **Precision Curve**:
   - Precision cao: Ít False Positive
   - Quan trọng khi cost of FP cao

4. **Recall Curve**:
   - Recall cao: Ít False Negative
   - Quan trọng trong intrusion detection (phải phát hiện được attack!)

**Ví dụ đọc biểu đồ**:
```
Epoch 1:  train_loss=0.50, val_loss=0.45 ✓ Good
Epoch 10: train_loss=0.20, val_loss=0.22 ✓ Good
Epoch 20: train_loss=0.10, val_loss=0.25 ⚠ Overfitting warning
Epoch 30: train_loss=0.05, val_loss=0.30 ✗ Overfitting!
```

---

### STEP 9: ĐÁNH GIÁ MÔ HÌNH

#### 9.1. Predictions

```python
y_pred_proba = model.predict(X_test_scaled, verbose=0)
# Output: [[0.3, 0.7], [0.9, 0.1], ...]
#          xác suất cho mỗi class

y_pred = np.argmax(y_pred_proba, axis=1)
# Lấy class có xác suất cao nhất
# [1, 0, 1, 0, ...]
```

#### 9.2. Metrics

**Accuracy**:
```python
accuracy = accuracy_score(y_test, y_pred)
# Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision**:
```python
precision = precision_score(y_test, y_pred, average='binary')
# Precision = TP / (TP + FP)
# "Trong các dự đoán Attack, bao nhiêu % thật sự là Attack?"
```

**Recall**:
```python
recall = recall_score(y_test, y_pred, average='binary')
# Recall = TP / (TP + FN)
# "Trong các Attack thực tế, bao nhiêu % được phát hiện?"
```

**F1-Score**:
```python
f1 = f1_score(y_test, y_pred, average='binary')
# F1 = 2 * (Precision * Recall) / (Precision + Recall)
# Trung bình điều hòa của Precision và Recall
```

#### 9.3. Confusion Matrix

```
                Predicted
                Benign  Attack
Actual Benign     TN      FP
       Attack     FN      TP
```

**Ví dụ cụ thể**:
```
                Predicted
                Benign    Attack
Actual Benign   3,800,000  200,000  (FP: False Alarm)
       Attack     300,000 4,700,000  (FN: Missed Attack)

TN = 3,800,000: Dự đoán đúng Benign
FP =   200,000: Dự đoán nhầm là Attack (False Alarm)
FN =   300,000: Dự đoán nhầm là Benign (Missed Attack) ⚠
TP = 4,700,000: Dự đoán đúng Attack
```

**Phân tích**:
- **FP (False Positive)**: Benign bị nhầm là Attack
  - Consequence: False alarm, block traffic bình thường
  - Ít nghiêm trọng hơn FN

- **FN (False Negative)**: Attack bị nhầm là Benign
  - Consequence: Attack không bị phát hiện!
  - Rất nghiêm trọng trong security!

**Trade-off**:
- Precision cao → FP thấp → Ít false alarm
- Recall cao → FN thấp → Ít missed attack
- Thường phải balance giữa hai metrics này

#### 9.4. Classification Report

```
              precision    recall  f1-score   support

      Benign     0.9267    0.9500    0.9382   4000000
      Attack     0.9592    0.9400    0.9495   5000000

    accuracy                         0.9444   9000000
   macro avg     0.9430    0.9450    0.9439   9000000
weighted avg     0.9447    0.9444    0.9445   9000000
```

**Giải thích**:
- **support**: Số samples thực tế của class đó
- **macro avg**: Trung bình đơn giản của 2 classes
- **weighted avg**: Trung bình có trọng số (theo support)

---

### STEP 10: LƯU KẾT QUẢ

#### 10.1. Lưu Model

```python
model.save('final_cnn_gru_model.h5')
```

**Giải thích**:
- Lưu toàn bộ model: architecture + weights + optimizer state
- Format: HDF5 (.h5)
- Có thể load lại để dùng: `model = keras.models.load_model('final_cnn_gru_model.h5')`

#### 10.2. Lưu Scaler và Label Encoder

```python
import joblib
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
```

**Tại sao phải lưu?**
- Khi dùng model để predict dữ liệu mới:
  1. Phải chuẩn hóa dữ liệu với **cùng scaler**
  2. Phải decode labels với **cùng label encoder**

**Ví dụ sử dụng**:
```python
# Load model và scaler
model = keras.models.load_model('final_cnn_gru_model.h5')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Predict dữ liệu mới
X_new = pd.read_csv('new_traffic.csv')
X_new_scaled = scaler.transform(X_new)  # Dùng scaler đã fit
predictions = model.predict(X_new_scaled)
labels = le.inverse_transform(predictions.argmax(axis=1))
print(labels)  # ['Benign', 'Attack', 'Attack', ...]
```

---

## 📊 ĐÁNH GIÁ VÀ PHÂN TÍCH

### Metrics Quan Trọng trong Intrusion Detection

#### 1. Recall (Sensitivity)
**Quan trọng nhất!**
- Phải phát hiện được càng nhiều attack càng tốt
- Recall thấp → Nhiều attack bị bỏ sót → Nguy hiểm!

#### 2. Precision
- Quan trọng để tránh false alarm
- Precision thấp → Nhiều traffic bình thường bị block → User experience kém

#### 3. F1-Score
- Balance giữa Precision và Recall
- Thường dùng để so sánh các models

#### 4. Accuracy
- **Cẩn thận với imbalanced dataset!**
- Ví dụ: 95% Benign, 5% Attack
  - Model dự đoán tất cả là Benign → Accuracy = 95%
  - Nhưng Recall = 0% → Vô dụng!

### So sánh với Baseline

**Baseline models thường dùng**:
- Logistic Regression
- Random Forest
- SVM
- Simple Neural Network

**Mục tiêu**:
- CNN-GRU phải tốt hơn baseline ít nhất 2-5%
- Trade-off giữa performance và complexity

---

## 🚀 TIPS VÀ TRICKS

### 1. Tăng Performance

**Tăng Recall** (phát hiện nhiều attack hơn):
- Giảm threshold của classification
  ```python
  threshold = 0.3  # instead of 0.5
  y_pred = (y_pred_proba[:, 1] > threshold).astype(int)
  ```
- Tăng class weight cho Attack
  ```python
  class_weight = {0: 1.0, 1: 2.0}  # Attack có weight gấp đôi
  model.fit(..., class_weight=class_weight)
  ```

**Giảm Overfitting**:
- Tăng Dropout rate (0.5 → 0.6, 0.7)
- Thêm L2 regularization
  ```python
  layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.01))
  ```
- Data augmentation
- Early stopping với patience nhỏ hơn

**Tăng Speed**:
- Giảm batch size (nhưng có thể giảm performance)
- Dùng GPU (CUDA)
- Mixed precision training (FP16)
- Model pruning/quantization

### 2. Debug Common Issues

**Problem: Loss không giảm**
- Check learning rate (quá cao hoặc quá thấp)
- Check data preprocessing (có normalize chưa?)
- Check label encoding (đúng format chưa?)

**Problem: Overfitting**
- Tăng Dropout
- Thêm regularization
- Giảm model complexity
- Tăng training data

**Problem: Underfitting**
- Tăng model complexity (thêm layers, units)
- Giảm regularization
- Train lâu hơn
- Check data quality

### 3. Hyperparameter Tuning

**Learning Rate**:
```python
# Try: 0.1, 0.01, 0.001, 0.0001
lr_options = [1e-2, 1e-3, 1e-4]
```

**Batch Size**:
```python
# Try: 32, 64, 128, 256, 512
batch_options = [64, 128, 256]
```

**Architecture**:
```python
# Try different:
# - Number of Conv blocks
# - Number of filters
# - GRU units
# - Dense layer sizes
```

---

## 📝 KẾT LUẬN

### Ưu điểm của CNN-GRU

1. **Kết hợp spatial và temporal features**
   - CNN: Local patterns trong features
   - GRU: Sequential patterns

2. **Performance cao**
   - Thường đạt 95-99% accuracy
   - Recall cao → Phát hiện attack tốt

3. **Robust**
   - Handle được imbalanced data
   - Generalize tốt

### Nhược điểm

1. **Computational cost**
   - Training lâu (vài giờ với dataset lớn)
   - Cần GPU để train hiệu quả

2. **Complex architecture**
   - Khó debug
   - Nhiều hyperparameters cần tune

3. **Black box**
   - Khó giải thích tại sao model dự đoán như vậy
   - Cần thêm explainability techniques

### Hướng phát triển

1. **Attention Mechanism**
   - Thêm attention layer để focus vào important features

2. **Ensemble Learning**
   - Kết hợp nhiều models
   - Voting hoặc stacking

3. **Real-time Detection**
   - Optimize cho inference speed
   - Deploy với TensorFlow Lite, ONNX

4. **Explainability**
   - SHAP, LIME để giải thích predictions
   - Feature importance analysis

---

## 📚 TÀI LIỆU THAM KHẢO

### Papers
- DeepFed Paper: Federated Learning Architecture
- CNN for Network Traffic Classification
- GRU vs LSTM comparison

### Libraries Documentation
- TensorFlow/Keras: https://www.tensorflow.org/
- Scikit-learn: https://scikit-learn.org/
- Pandas: https://pandas.pydata.org/

### Courses
- Deep Learning Specialization (Coursera)
- TensorFlow Developer Certificate
- Network Security và Intrusion Detection

---

## 💡 FAQ - Câu hỏi thường gặp

### Q1: Tại sao dùng binary classification thay vì multi-class?
**A**: 
- Đơn giản hơn, dễ deploy hơn
- Performance thường tốt hơn
- Trong thực tế, quan trọng nhất là phát hiện "có attack hay không"
- Chi tiết loại attack có thể phát hiện ở stage 2

### Q2: Dataset quá lớn, không đủ RAM?
**A**:
- Dùng `batch_size` nhỏ hơn
- Dùng generator thay vì load toàn bộ:
  ```python
  def data_generator(X, y, batch_size):
      while True:
          for i in range(0, len(X), batch_size):
              yield X[i:i+batch_size], y[i:i+batch_size]
  ```
- Downsample dataset (lấy subset)
- Dùng cloud computing (AWS, GCP)

### Q3: Training quá lâu?
**A**:
- Dùng GPU (NVIDIA CUDA)
- Giảm số epochs
- Giảm batch size
- Simplify model architecture
- Dùng pretrained model

### Q4: Làm sao improve Recall?
**A**:
- Adjust classification threshold
- Class weighting
- Oversample minority class (SMOTE)
- Focal loss thay vì cross-entropy
- Ensemble với nhiều models

### Q5: Model bị overfitting?
**A**:
- Tăng Dropout (0.5 → 0.7)
- L2 regularization
- Early stopping với patience nhỏ
- Data augmentation
- Reduce model complexity

---

**Chúc bạn thành công với dự án! 🎉**

*Nếu có câu hỏi, vui lòng tham khảo documentation hoặc contact!*
