# BÁO CÁO TOÀN DIỆN: MÔ HÌNH CNN-GRU CHO PHÁT HIỆN XÂM NHẬP IoT

## 📋 MỤC LỤC

1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Kiến Trúc Mô Hình](#2-kiến-trúc-mô-hình)
3. [Chi Tiết Các Module](#3-chi-tiết-các-module)
4. [Công Thức Toán Học](#4-công-thức-toán-học)
5. [Quá Trình Tiền Xử Lý Dữ Liệu](#5-quá-trình-tiền-xử-lý-dữ-liệu)
6. [Thuật Toán Huấn Luyện](#6-thuật-toán-huấn-luyện)
7. [Xử Lý Imbalanced Data](#7-xử-lý-imbalanced-data)
8. [Callbacks và Optimization](#8-callbacks-và-optimization)
9. [Đánh Giá Mô Hình](#9-đánh-giá-mô-hình)
10. [Kết Luận](#10-kết-luận)

---

## 1. TỔNG QUAN HỆ THỐNG

### 1.1. Mục Đích
Xây dựng hệ thống phát hiện xâm nhập (Intrusion Detection System - IDS) cho môi trường IoT sử dụng Deep Learning, dựa trên kiến trúc **DeepFed**.

### 1.2. Bài Toán
- **Loại bài toán**: Binary Classification (Phân loại nhị phân)
- **Input**: Network traffic features (39 features)
- **Output**: 2 classes
  - Class 0: **Attack** (tấn công)
  - Class 1: **Benign** (lưu lượng bình thường)

### 1.3. Dataset
- **Tên**: IoT Dataset 2023
- **Tổng số mẫu**: ~15 triệu samples
- **Số features**: 39 features (sau khi loại bỏ constant columns)
- **Phân bố**:
  - Attack: 14,725,951 mẫu (97.66%)
  - Benign: 352,302 mẫu (2.34%)
  - **Imbalance Ratio**: 41.80:1

### 1.4. Chia Dữ Liệu
```
Training Set:   70% (~10.5 triệu mẫu)
Validation Set: 10% (~1.5 triệu mẫu)
Test Set:       20% (~3.0 triệu mẫu)
```

---

## 2. KIẾN TRÚC MÔ HÌNH

### 2.1. Sơ Đồ Tổng Quan

```
                        INPUT (39 features)
                              |
                    +---------+---------+
                    |                   |
               [CNN Module]        [GRU Module]
                    |                   |
              Conv1D -> BN          GRU Layer 1
              MaxPool              (128 units)
                    |                   |
              Conv1D -> BN          GRU Layer 2
              MaxPool              (64 units)
                    |                   |
              Conv1D -> BN              |
              MaxPool                   |
                    |                   |
                 Flatten                |
                    |                   |
                    +--------+----------+
                             |
                      [Concatenate]
                             |
                      [MLP Module]
                             |
                    Dense(256) -> BN -> Dropout(0.5)
                             |
                    Dense(128) -> BN -> Dropout(0.3)
                             |
                     Dense(2) + Softmax
                             |
                    OUTPUT (2 classes)
```

### 2.2. Thông Số Mô Hình
- **Tổng số parameters**: 526,338 parameters (~2 MB)
- **Trainable parameters**: 524,674
- **Non-trainable parameters**: 1,664 (BatchNormalization)

---

## 3. CHI TIẾT CÁC MODULE

### 3.1. CNN Module (Convolutional Neural Network)

**Mục đích**: Trích xuất **spatial features** (đặc trưng không gian) từ dữ liệu.

#### Cấu trúc:
```python
Input: (batch_size, 39)
  ↓
Reshape: (batch_size, 39, 1)  # Chuyển thành dạng time series
  ↓
Conv Block 1:
  - Conv1D(64 filters, kernel=3, activation=ReLU)
  - BatchNormalization
  - MaxPooling1D(pool_size=2)
  Output: (batch_size, 19, 64)
  ↓
Conv Block 2:
  - Conv1D(128 filters, kernel=3, activation=ReLU)
  - BatchNormalization
  - MaxPooling1D(pool_size=2)
  Output: (batch_size, 9, 128)
  ↓
Conv Block 3:
  - Conv1D(256 filters, kernel=3, activation=ReLU)
  - BatchNormalization
  - MaxPooling1D(pool_size=2)
  Output: (batch_size, 4, 256)
  ↓
Flatten: (batch_size, 1024)
```

#### Vai trò từng thành phần:

**Conv1D (Convolutional 1D):**
- Quét qua chuỗi features với sliding window
- Phát hiện patterns cục bộ trong dữ liệu
- Mỗi filter học một pattern khác nhau

**BatchNormalization:**
- Chuẩn hóa output của mỗi layer
- Tăng tốc độ training
- Giảm Internal Covariate Shift

**MaxPooling:**
- Giảm kích thước dữ liệu (downsampling)
- Giữ lại features quan trọng nhất
- Tạo translation invariance

---

### 3.2. GRU Module (Gated Recurrent Unit)

**Mục đích**: Trích xuất **temporal features** (đặc trưng thời gian) và phụ thuộc tuần tự.

#### Cấu trúc:
```python
Input: (batch_size, 39)
  ↓
Reshape: (batch_size, 39, 1)
  ↓
GRU Layer 1:
  - 128 units
  - return_sequences=True
  Output: (batch_size, 39, 128)
  ↓
GRU Layer 2:
  - 64 units
  - return_sequences=False
  Output: (batch_size, 64)
```

#### Tại sao chọn GRU thay vì LSTM?
- **GRU nhanh hơn**: Ít parameters hơn LSTM (2 gates vs 3 gates)
- **Hiệu quả tương đương**: Với dataset lớn, GRU cho kết quả gần như LSTM
- **Tránh overfitting**: Ít parameters = giảm risk overfitting

---

### 3.3. MLP Module (Multi-Layer Perceptron)

**Mục đích**: Kết hợp features từ CNN và GRU, thực hiện classification.

#### Cấu trúc:
```python
Input: Concatenate([CNN_output, GRU_output])
       Shape: (batch_size, 1024 + 64 = 1088)
  ↓
Dense Layer 1:
  - 256 units, activation=ReLU
  - BatchNormalization
  - Dropout(0.5)
  Output: (batch_size, 256)
  ↓
Dense Layer 2:
  - 128 units, activation=ReLU
  - BatchNormalization
  - Dropout(0.3)
  Output: (batch_size, 128)
  ↓
Output Layer:
  - 2 units, activation=Softmax
  Output: (batch_size, 2)  # [P(Attack), P(Benign)]
```

#### Vai trò Dropout:
- **Dropout(0.5)**: Tắt ngẫu nhiên 50% neurons trong training
- **Dropout(0.3)**: Tắt ngẫu nhiên 30% neurons
- **Mục đích**: Tránh overfitting, tăng generalization

---

## 4. CÔNG THỨC TOÁN HỌC

### 4.1. Convolutional Layer

**Công thức Conv1D:**
```
y[i] = σ(Σ(w[k] * x[i+k]) + b)

Trong đó:
- x: input sequence
- w: filter weights (kernel)
- b: bias
- σ: activation function (ReLU)
- k: kernel size
```

**ReLU Activation:**
```
ReLU(x) = max(0, x)
```

**Ưu điểm ReLU:**
- Tính toán nhanh
- Giảm vanishing gradient
- Tạo sparsity (nhiều neurons = 0)

---

### 4.2. Batch Normalization

**Công thức:**
```
Step 1: Tính mean và variance của mini-batch
μ_B = (1/m) * Σ(x_i)
σ²_B = (1/m) * Σ(x_i - μ_B)²

Step 2: Normalize
x̂_i = (x_i - μ_B) / √(σ²_B + ε)

Step 3: Scale và shift
y_i = γ * x̂_i + β

Trong đó:
- μ_B: mean của batch
- σ²_B: variance của batch
- ε: số nhỏ tránh chia cho 0 (thường = 1e-5)
- γ, β: learnable parameters
```

**Lợi ích:**
- Ổn định quá trình training
- Cho phép learning rate cao hơn
- Giảm phụ thuộc vào initialization

---

### 4.3. MaxPooling

**Công thức:**
```
y[i] = max(x[i*stride : i*stride + pool_size])

Ví dụ với pool_size=2:
Input:  [3, 7, 2, 9, 4, 6]
Output: [7, 9, 6]
```

**Ưu điểm:**
- Giảm computational cost
- Tạo invariance to small translations
- Giữ lại features mạnh nhất

---

### 4.4. GRU (Gated Recurrent Unit)

**Công thức GRU:**

GRU có 2 gates: **Reset Gate** và **Update Gate**

```
1. Reset Gate (r_t):
   r_t = σ(W_r · [h_{t-1}, x_t] + b_r)

2. Update Gate (z_t):
   z_t = σ(W_z · [h_{t-1}, x_t] + b_z)

3. Candidate Hidden State (h̃_t):
   h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)

4. Final Hidden State (h_t):
   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

Trong đó:
- x_t: input tại thời điểm t
- h_t: hidden state tại thời điểm t
- σ: sigmoid function
- ⊙: element-wise multiplication
- W, b: learnable weights và biases
```

**Giải thích:**

1. **Reset Gate (r_t)**: Quyết định bỏ qua bao nhiêu thông tin từ quá khứ
   - r_t ≈ 0: Bỏ qua hết thông tin cũ
   - r_t ≈ 1: Giữ lại toàn bộ thông tin cũ

2. **Update Gate (z_t)**: Quyết định cập nhật bao nhiêu thông tin mới
   - z_t ≈ 0: Giữ nguyên h_{t-1}
   - z_t ≈ 1: Thay thế hoàn toàn bằng h̃_t

3. **Candidate (h̃_t)**: Thông tin mới được tính toán

4. **Final State (h_t)**: Tổ hợp giữa thông tin cũ và mới

---

### 4.5. Dense Layer (Fully Connected)

**Công thức:**
```
y = σ(W · x + b)

Trong đó:
- x: input vector (shape: n)
- W: weight matrix (shape: m × n)
- b: bias vector (shape: m)
- σ: activation function
- y: output vector (shape: m)
```

**Matrix Multiplication:**
```
y[i] = Σ(W[i,j] * x[j]) + b[i]  for j = 1 to n
```

---

### 4.6. Dropout

**Công thức (Training):**
```
y[i] = {
    0              with probability p
    x[i]/(1-p)     with probability (1-p)
}

Ví dụ với p=0.5 (Dropout 50%):
Input:  [2, 4, 6, 8]
Mask:   [1, 0, 1, 0]  (random)
Output: [4, 0, 12, 0]  (scaled by 1/(1-0.5)=2)
```

**Inference (Testing):**
```
y[i] = x[i]  (không dropout)
```

**Tại sao scale bằng 1/(1-p)?**
- Để đảm bảo expected value giống nhau giữa training và testing
- Training: E[y] = x * (1-p) * 1/(1-p) = x
- Testing: E[y] = x

---

### 4.7. Softmax Activation

**Công thức:**
```
softmax(x_i) = exp(x_i) / Σ(exp(x_j)) for j = 1 to n

Ví dụ:
Input:  [2.0, 1.0, 0.1]
Output: [0.659, 0.242, 0.099]
```

**Đặc điểm:**
- Output luôn trong khoảng [0, 1]
- Tổng các output = 1 (phân bố xác suất)
- Thích hợp cho multi-class classification

---

### 4.8. Loss Function: Sparse Categorical Crossentropy

**Công thức:**
```
Loss = -log(p_{y_true})

Trong đó:
- y_true: true label (0 hoặc 1)
- p_{y_true}: xác suất dự đoán cho class đúng

Ví dụ:
True label: 1 (Benign)
Predictions: [0.3, 0.7]  # [P(Attack), P(Benign)]
Loss = -log(0.7) = 0.357
```

**Với Class Weights:**
```
Weighted_Loss = w_{y_true} * (-log(p_{y_true}))

Trong đó:
- w_{y_true}: weight của class đúng
```

**Tổng Loss cho toàn bộ dataset:**
```
Total_Loss = (1/N) * Σ(Loss_i) for i = 1 to N

Với Class Weights:
Total_Loss = (1/Σw_i) * Σ(w_i * Loss_i)
```

---

### 4.9. Optimizer: Adam

**Công thức Adam (Adaptive Moment Estimation):**

```
Step 1: Tính gradient
g_t = ∇L(θ_{t-1})

Step 2: Tính first moment (momentum)
m_t = β_1 * m_{t-1} + (1 - β_1) * g_t

Step 3: Tính second moment (RMSprop)
v_t = β_2 * v_{t-1} + (1 - β_2) * g_t²

Step 4: Bias correction
m̂_t = m_t / (1 - β_1^t)
v̂_t = v_t / (1 - β_2^t)

Step 5: Update parameters
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)

Hyperparameters (mặc định):
- α (learning rate): 0.001
- β_1: 0.9
- β_2: 0.999
- ε: 1e-7
```

**Tại sao chọn Adam?**
- Kết hợp momentum và RMSprop
- Adaptive learning rate cho từng parameter
- Hoạt động tốt với sparse gradients
- Ít cần tune hyperparameters

---

## 5. QUÁ TRÌNH TIỀN XỬ LÝ DỮ LIỆU

### 5.1. Load và Merge Data

```python
# Load 20 CSV files
for file in csv_files:
    df_temp = pd.read_csv(file)
    dfs.append(df_temp)

# Merge tất cả
df = pd.concat(dfs, ignore_index=True)
```

---

### 5.2. Label Mapping

**Chuyển đổi Multi-class → Binary:**

```python
def map_to_binary(label):
    if 'benign' in label.lower():
        return 'Benign'
    else:
        return 'Attack'

# Áp dụng
df['binary_label'] = df['Label'].apply(map_to_binary)

# Encode thành số
LabelEncoder:
  'Attack' → 0
  'Benign' → 1
```

**Attack types bao gồm:**
- DDOS attacks (ICMP, UDP, TCP, SYN flood, ...)
- DOS attacks
- MIRAI botnet
- Scanning attacks (Port scan, OS scan, ...)
- Web attacks (SQL injection, XSS, ...)
- MITM attacks

---

### 5.3. Xử Lý Missing Values

```python
# Kiểm tra missing values
missing = X.isnull().sum().sum()

# Fill với 0
if missing > 0:
    X = X.fillna(0)
```

**Tại sao fill bằng 0?**
- 0 là giá trị neutral trong network traffic
- Không làm sai lệch phân bố sau normalization

---

### 5.4. Xử Lý Infinite Values

```python
# Thay thế inf và -inf
X = X.replace([np.inf, -np.inf], 0)
```

**Nguồn gốc infinite values:**
- Chia cho 0 trong feature engineering
- Log của số âm hoặc 0
- Overflow trong tính toán

---

### 5.5. Loại Bỏ Constant Columns

```python
# Tìm các cột có duy nhất 1 giá trị
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]

# Loại bỏ
X = X.drop(constant_cols, axis=1)
```

**Lý do:**
- Constant columns không mang thông tin
- Gây lãng phí computational resources

---

### 5.6. Normalization: StandardScaler

**Công thức:**
```
x_scaled = (x - μ) / σ

Trong đó:
- μ: mean của feature
- σ: standard deviation của feature
```

**Ví dụ:**
```
Original: [100, 200, 300, 400, 500]
μ = 300
σ = 141.42

Scaled: [-1.414, -0.707, 0, 0.707, 1.414]
```

**Lợi ích:**
- Mean = 0, Std = 1
- Giúp gradient descent hội tụ nhanh hơn
- Tránh features có range lớn "dominate" model

**Quan trọng:**
```python
# Fit trên training set
scaler.fit(X_train)

# Transform cả 3 tập
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)    # Dùng statistics từ train
X_test_scaled = scaler.transform(X_test)  # Dùng statistics từ train
```

**Tại sao không fit trên val/test?**
- Tránh **data leakage**
- Model phải học từ training distribution
- Testing phải mô phỏng real-world (không biết trước statistics)

---

## 6. THUẬT TOÁN HUẤN LUYỆN

### 6.1. Forward Propagation

**Quá trình:**

```
Step 1: Input → CNN Module
  x → Reshape → Conv1D → BN → Pool → ... → Flatten
  Output: CNN_features (1024 dimensions)

Step 2: Input → GRU Module
  x → Reshape → GRU1 → GRU2
  Output: GRU_features (64 dimensions)

Step 3: Concatenate
  Combined = [CNN_features, GRU_features]
  Output: (1088 dimensions)

Step 4: MLP Module
  Combined → Dense → BN → Dropout → Dense → BN → Dropout
  Output: Hidden_features (128 dimensions)

Step 5: Classification
  Hidden → Dense(2) → Softmax
  Output: [P(Attack), P(Benign)]
```

---

### 6.2. Loss Calculation

**Với Class Weights:**

```python
# Giả sử batch có 4 samples:
y_true = [0, 1, 0, 0]  # Attack, Benign, Attack, Attack
y_pred = [[0.9, 0.1],  # Dự đoán đúng Attack
          [0.3, 0.7],  # Dự đoán đúng Benign
          [0.8, 0.2],  # Dự đoán đúng Attack
          [0.6, 0.4]]  # Dự đoán đúng Attack

# Class weights (ví dụ)
w_attack = 0.024
w_benign = 1.0

# Loss cho từng sample
loss[0] = w_attack * (-log(0.9)) = 0.024 * 0.105 = 0.00252
loss[1] = w_benign * (-log(0.7)) = 1.0 * 0.357 = 0.357
loss[2] = w_attack * (-log(0.8)) = 0.024 * 0.223 = 0.00535
loss[3] = w_attack * (-log(0.6)) = 0.024 * 0.511 = 0.01226

# Total loss
Total_Loss = mean(loss) = 0.094
```

**Nhận xét:**
- Sample Benign đóng góp loss lớn hơn (~97% total loss)
- Model sẽ focus nhiều hơn vào class Benign
- Giải quyết imbalance problem

---

### 6.3. Backward Propagation

**Công thức Chain Rule:**

```
∂Loss/∂W = ∂Loss/∂y * ∂y/∂z * ∂z/∂W

Trong đó:
- y: output của layer
- z: input của activation function
- W: weights
```

**Ví dụ với Dense Layer:**

```
Layer: z = W·x + b
Activation: y = ReLU(z)
Loss: L

∂L/∂W = ∂L/∂y * ∂y/∂z * ∂z/∂W
      = ∂L/∂y * ReLU'(z) * x

Trong đó:
ReLU'(z) = {
    1  if z > 0
    0  if z ≤ 0
}
```

---

### 6.4. Weight Update với Adam

```python
# Pseudo-code
for epoch in range(EPOCHS):
    for batch in training_data:
        # Forward pass
        predictions = model(batch_X)

        # Calculate loss với class weights
        loss = weighted_crossentropy(predictions, batch_y, class_weights)

        # Backward pass
        gradients = compute_gradients(loss)

        # Update weights với Adam
        optimizer.apply_gradients(gradients)
```

---

### 6.5. Training Loop Chi Tiết

```python
# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 2048
LEARNING_RATE = 0.001

# Mỗi epoch
for epoch in range(EPOCHS):
    # Training phase
    for batch_idx in range(num_batches):
        # Lấy batch
        batch_X = X_train[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]
        batch_y = y_train[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]

        # Forward + Backward + Update
        # (được handle bởi model.fit())

    # Validation phase
    val_loss = evaluate(X_val, y_val)
    val_accuracy = evaluate_accuracy(X_val, y_val)

    # Callbacks
    # - EarlyStopping: kiểm tra val_loss
    # - ReduceLROnPlateau: giảm learning rate nếu cần
    # - ModelCheckpoint: lưu best model
```

---

## 7. XỬ LÝ IMBALANCED DATA

### 7.1. Vấn Đề Imbalance

**Phân bố dataset:**
```
Attack: 14,725,951 (97.66%)
Benign:    352,302 (2.34%)
Ratio: 41.80:1
```

**Hệ quả nếu không xử lý:**
- Model sẽ bias về class Attack
- Dự đoán "tất cả là Attack" → accuracy 97.66% nhưng vô dụng!
- Recall của Benign rất thấp (nhiều False Negatives)

---

### 7.2. Giải Pháp: Class Weights

**Công thức tính Class Weight:**

```
w_i = n_samples / (n_classes * n_samples_i)

Trong đó:
- n_samples: tổng số samples
- n_classes: số lượng classes
- n_samples_i: số samples của class i
```

**Áp dụng:**

```python
from sklearn.utils.class_weight import compute_class_weight

# Tính toán
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Kết quả (ví dụ):
# class 0 (Attack): w = 0.024
# class 1 (Benign): w = 1.000

# Benign được tăng trọng số ~42 lần so với Attack
```

**Cách hoạt động:**

```
Sample Attack có loss = 0.1
  → Weighted loss = 0.024 * 0.1 = 0.0024

Sample Benign có loss = 0.1
  → Weighted loss = 1.0 * 0.1 = 0.1

→ Model quan tâm nhiều hơn đến Benign!
```

---

### 7.3. Ảnh Hưởng Đến Training

**Không có Class Weights:**
```
Epoch 1: Accuracy = 97.5%, Recall(Benign) = 10%
Epoch 10: Accuracy = 98.0%, Recall(Benign) = 20%
→ Model chỉ học predict "Attack"
```

**Có Class Weights:**
```
Epoch 1: Accuracy = 95.0%, Recall(Benign) = 60%
Epoch 10: Accuracy = 98.5%, Recall(Benign) = 95%
→ Model học cân bằng cả 2 classes
```

---

## 8. CALLBACKS VÀ OPTIMIZATION

### 8.1. EarlyStopping

**Mục đích**: Dừng training khi model không còn cải thiện

**Cơ chế:**
```python
EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

**Hoạt động:**

```
Epoch 1:  val_loss = 0.100 → Best = 0.100, Counter = 0
Epoch 2:  val_loss = 0.090 → Best = 0.090, Counter = 0
Epoch 3:  val_loss = 0.085 → Best = 0.085, Counter = 0
...
Epoch 15: val_loss = 0.050 → Best = 0.050, Counter = 0
Epoch 16: val_loss = 0.051 → Best = 0.050, Counter = 1
Epoch 17: val_loss = 0.052 → Best = 0.050, Counter = 2
...
Epoch 25: val_loss = 0.055 → Best = 0.050, Counter = 10
→ STOP! Restore weights từ Epoch 15
```

**Lợi ích:**
- Tránh overfitting
- Tiết kiệm thời gian training
- Tự động chọn số epochs tối ưu

---

### 8.2. ReduceLROnPlateau

**Mục đích**: Giảm learning rate khi model plateau (không cải thiện)

**Cơ chế:**
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)
```

**Hoạt động:**

```
Epoch 1-10:  val_loss giảm → lr = 0.001
Epoch 11-15: val_loss không đổi → Counter = 5 → lr = 0.001 * 0.5 = 0.0005
Epoch 16-20: val_loss giảm tiếp → lr = 0.0005
Epoch 21-25: val_loss không đổi → Counter = 5 → lr = 0.0005 * 0.5 = 0.00025
...
```

**Lý do:**
- Learning rate cao: Hội tụ nhanh nhưng có thể bỏ qua optimum
- Learning rate thấp: Hội tụ chậm nhưng chính xác hơn
- Adaptive LR: Kết hợp ưu điểm cả 2

---

### 8.3. ModelCheckpoint

**Mục đích**: Lưu model tốt nhất trong quá trình training

**Cơ chế:**
```python
ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
```

**Hoạt động:**

```
Epoch 1:  val_acc = 0.95 → Save (best so far)
Epoch 2:  val_acc = 0.96 → Save (better)
Epoch 3:  val_acc = 0.95 → Skip (worse)
Epoch 4:  val_acc = 0.97 → Save (better)
...
Epoch 50: val_acc = 0.96 → Skip

→ Final model = Epoch 4 model (val_acc = 0.97)
```

**Lợi ích:**
- Không lo mất model tốt nhất nếu training bị lỗi
- Tự động chọn model perform tốt nhất
- Backup an toàn

---

## 9. ĐÁNH GIÁ MÔ HÌNH

### 9.1. Confusion Matrix

**Định nghĩa:**

```
                    Predicted
                Attack    Benign
Actual  Attack     TP        FN
        Benign     FP        TN

TP: True Positive  - Dự đoán Attack, thực tế Attack ✓
TN: True Negative  - Dự đoán Benign, thực tế Benign ✓
FP: False Positive - Dự đoán Attack, thực tế Benign ✗ (Type I Error)
FN: False Negative - Dự đoán Benign, thực tế Attack ✗ (Type II Error)
```

---

### 9.2. Metrics

**1. Accuracy (Độ chính xác tổng thể)**

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Ý nghĩa: Tỷ lệ dự đoán đúng trên tổng số mẫu
Phạm vi: [0, 1], cao hơn = tốt hơn
```

**Ví dụ:**
```
TP=2900000, TN=68000, FP=2000, FN=13000
Accuracy = (2900000 + 68000) / 3000000 = 0.995 (99.5%)
```

---

**2. Precision (Độ chính xác dương tính)**

```
Precision = TP / (TP + FP)

Ý nghĩa: Trong các mẫu dự đoán là Attack, có bao nhiêu % đúng?
Câu hỏi: "Khi model báo Attack, tôi tin được bao nhiêu?"
```

**Ví dụ:**
```
TP=2900000, FP=2000
Precision = 2900000 / (2900000 + 2000) = 0.9993 (99.93%)
```

---

**3. Recall (Độ nhạy, Sensitivity, True Positive Rate)**

```
Recall = TP / (TP + FN)

Ý nghĩa: Trong các mẫu thực tế là Attack, model phát hiện được bao nhiêu %?
Câu hỏi: "Model bỏ sót bao nhiêu attacks?"
```

**Ví dụ:**
```
TP=2900000, FN=13000
Recall = 2900000 / (2900000 + 13000) = 0.9955 (99.55%)
```

**Quan trọng trong IDS:**
- Recall thấp = Nhiều attacks bị bỏ sót = Nguy hiểm!
- Trong IDS, Recall quan trọng hơn Precision

---

**4. F1-Score (Harmonic Mean của Precision và Recall)**

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)

Ý nghĩa: Điểm cân bằng giữa Precision và Recall
Phạm vi: [0, 1], cao hơn = tốt hơn
```

**Ví dụ:**
```
Precision = 0.9993, Recall = 0.9955
F1 = 2 * (0.9993 * 0.9955) / (0.9993 + 0.9955) = 0.9974 (99.74%)
```

**Tại sao dùng Harmonic Mean?**
```
Arithmetic Mean = (0.9993 + 0.9955) / 2 = 0.9974
Harmonic Mean = 2 * (0.9993 * 0.9955) / (0.9993 + 0.9955) = 0.9974

Nhưng với số liệu khác:
Precision = 1.0, Recall = 0.1
Arithmetic = 0.55 (misleading!)
Harmonic = 0.18 (phản ánh đúng model kém)
```

---

### 9.3. Đánh Giá Cho Binary Classification

**Đối với class Attack (class 0):**
```
Precision_Attack = TP_attack / (TP_attack + FP_attack)
Recall_Attack = TP_attack / (TP_attack + FN_attack)
F1_Attack = 2 * P * R / (P + R)
```

**Đối với class Benign (class 1):**
```
Precision_Benign = TN / (TN + FN)
Recall_Benign = TN / (TN + FP)
F1_Benign = 2 * P * R / (P + R)
```

**Overall Metrics:**
```
Macro-average: Trung bình không trọng số
  F1_macro = (F1_Attack + F1_Benign) / 2

Weighted-average: Trung bình có trọng số
  F1_weighted = (n_attack * F1_Attack + n_benign * F1_Benign) / n_total
```

---

## 10. KẾT LUẬN

### 10.1. Ưu Điểm Của Kiến Trúc CNN-GRU

**1. Kết hợp 2 loại features:**
- CNN: Spatial patterns (đặc trưng cục bộ)
- GRU: Temporal patterns (đặc trưng thời gian)

**2. Hiệu quả với dữ liệu IoT:**
- Network traffic có cả spatial và temporal characteristics
- CNN phát hiện attack signatures
- GRU phát hiện attack sequences

**3. Performance cao:**
- Accuracy > 99%
- Recall > 98% (ít bỏ sót attacks)
- Training time hợp lý với GPU

---

### 10.2. Các Kỹ Thuật Quan Trọng

**1. Xử lý Imbalanced Data:**
- Class Weights → Giải quyết ratio 42:1
- Không cần SMOTE/undersampling

**2. Regularization:**
- BatchNormalization → Ổn định training
- Dropout (0.5, 0.3) → Tránh overfitting
- EarlyStopping → Dừng đúng lúc

**3. Optimization:**
- Adam optimizer → Adaptive learning rate
- ReduceLROnPlateau → Fine-tuning
- Class Weights → Focus vào minority class

---

### 10.3. Điểm Mạnh So Với Các Phương Pháp Khác

**So với Traditional ML (Random Forest, SVM):**
- ✓ Tự động feature extraction
- ✓ Xử lý được dữ liệu phức tạp
- ✓ Capture được temporal dependencies

**So với Simple DNN:**
- ✓ CNN giảm số parameters
- ✓ GRU xử lý sequences tốt hơn
- ✓ Ít overfitting hơn

**So với LSTM:**
- ✓ GRU nhanh hơn (ít parameters)
- ✓ Hiệu quả tương đương với dataset lớn
- ✓ Dễ train hơn

---

### 10.4. Khuyến Nghị

**Để cải thiện thêm:**

1. **Feature Engineering:**
   - Thêm domain-specific features
   - Feature selection để giảm dimensions

2. **Model Ensemble:**
   - Kết hợp nhiều models
   - Voting hoặc stacking

3. **Hyperparameter Tuning:**
   - Grid Search / Random Search
   - Bayesian Optimization

4. **Data Augmentation:**
   - Synthetic minority oversampling
   - Adversarial training

---

### 10.5. Workflow Tổng Quát

```
1. DATA PREPARATION
   ├─ Load CSV files
   ├─ Merge datasets
   ├─ Handle missing/infinite values
   ├─ Binary label mapping
   └─ Train/Val/Test split (70/10/20)

2. PREPROCESSING
   ├─ Remove constant columns
   ├─ StandardScaler normalization
   └─ Calculate class weights

3. MODEL BUILDING
   ├─ CNN Module (3 Conv blocks)
   ├─ GRU Module (2 GRU layers)
   ├─ Concatenate
   └─ MLP Module (2 Dense + Dropout)

4. TRAINING
   ├─ Adam optimizer (lr=0.001)
   ├─ Sparse Categorical Crossentropy + Class Weights
   ├─ Callbacks: EarlyStopping, ReduceLR, Checkpoint
   └─ 50 epochs, batch_size=2048

5. EVALUATION
   ├─ Confusion Matrix
   ├─ Accuracy, Precision, Recall, F1-Score
   └─ Classification Report

6. DEPLOYMENT
   ├─ Save model (.h5)
   ├─ Save scaler (.pkl)
   ├─ Save label encoder (.pkl)
   └─ Documentation
```

---

## PHỤ LỤC: BẢNG TÓM TẮT HYPERPARAMETERS

| Component | Hyperparameter | Value | Lý do chọn |
|-----------|---------------|-------|------------|
| **CNN** | Filters | 64, 128, 256 | Tăng dần để capture complex patterns |
| | Kernel size | 3 | Cân bằng receptive field và computation |
| | Pooling | MaxPool(2) | Giảm dimensions 50% |
| **GRU** | Units | 128, 64 | Đủ lớn để capture temporal dependencies |
| | Layers | 2 | Cân bằng capacity và overfitting |
| **MLP** | Units | 256, 128 | Giảm dần để extract high-level features |
| | Dropout | 0.5, 0.3 | Regularization mạnh ở layer đầu |
| **Training** | Epochs | 50 | Theo paper DeepFed |
| | Batch size | 2048 | Tối ưu cho GPU 8GB |
| | Learning rate | 0.001 | Adam default, hiệu quả với dataset lớn |
| **Callbacks** | EarlyStopping patience | 10 | Cho phép recover từ temporary plateaus |
| | ReduceLR patience | 5 | Nhanh hơn EarlyStopping |
| | ReduceLR factor | 0.5 | Giảm LR 50% mỗi lần |

---

## TÀI LIỆU THAM KHẢO

1. **DeepFed Paper**: Federated Learning Architecture for IoT IDS
2. **GRU Paper**: Cho et al. (2014) - "Learning Phrase Representations using RNN Encoder-Decoder"
3. **Adam Optimizer**: Kingma & Ba (2014) - "Adam: A Method for Stochastic Optimization"
4. **Batch Normalization**: Ioffe & Szegedy (2015) - "Batch Normalization: Accelerating Deep Network Training"
5. **Dropout**: Srivastava et al. (2014) - "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"

---

**© 2025 - IoT Intrusion Detection System**
**CNN-GRU Deep Learning Model**
