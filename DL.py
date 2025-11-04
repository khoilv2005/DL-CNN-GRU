import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ← FIX: Dùng backend không cần GUI (quan trọng cho WSL!)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib  # ← FIX: Import ở đầu file thay vì giữa code
import os
import warnings
import shutil
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 100)
print(" " * 30 + "CNN-GRU MODEL FOR IoT INTRUSION DETECTION")
print(" " * 35 + "Based on DeepFed Paper Architecture")
print("=" * 100)

# Check GPU availability
print("\n" + "=" * 100)
print("KIỂM TRA THIẾT BỊ TÍNH TOÁN")
print("=" * 100)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n✓ TÌM THẤY {len(gpus)} GPU!")
    for i, gpu in enumerate(gpus):
        gpu_details = tf.config.experimental.get_device_details(gpu)
        gpu_name = gpu_details.get('device_name', 'Unknown GPU')
        print(f"  GPU {i}: {gpu_name}")
        print(f"  Device: {gpu.name}")
    print(f"\n→ MODEL SẼ TRAIN TRÊN GPU (nhanh hơn 5-10 lần)")
else:
    print("\n✗ KHÔNG TÌM THẤY GPU")
    print("→ Model sẽ train trên CPU (chậm hơn)")

print("=" * 100)

# ================================================================================
# 0. CREATE BACKUP FOLDER
# ================================================================================

# Tạo thư mục backup với timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_FOLDER = f"./backup_{TIMESTAMP}"
os.makedirs(BACKUP_FOLDER, exist_ok=True)

print("\n" + "=" * 100)
print("BACKUP FOLDER")
print("=" * 100)
print(f"\n✓ Tạo thư mục backup: {BACKUP_FOLDER}")
print(f"  → Tất cả kết quả sẽ được lưu tại đây")
print("=" * 100)

# ================================================================================
# 1. LOAD AND ANALYZE DATASET
# ================================================================================

print("\n" + "=" * 100)
print("STEP 1: LOAD AND ANALYZE DATASET")
print("=" * 100)

DATA_PATH = './IoT_Dataset_2023'  # Thay đổi path của bạn

# Tìm tất cả file CSV
csv_files = []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))

# Sử dụng 20 FILES để có đủ data (theo paper DeepFed cần dataset lớn)
csv_files = sorted(csv_files)[:20]  # Lấy 20 files (~14-15M samples)

print(f"\nTìm thấy {len(csv_files)} file CSV (sử dụng 20 files để có đủ data)")
print("-" * 100)

# Load tất cả files
dfs = []
total_loaded = 0
for file in csv_files:
    try:
        df_temp = pd.read_csv(file)
        dfs.append(df_temp)
        total_loaded += 1
        print(f"✓ Loaded: {os.path.basename(file):50s} - {len(df_temp):>10,} samples")
    except MemoryError:
        print(f"\n⚠️  CẢNH BÁO: HẾT RAM khi load file {os.path.basename(file)}")
        print(f"→ Đã load được {total_loaded}/{len(csv_files)} files. Tiếp tục với {total_loaded} files...")
        break
    except Exception as e:
        print(f"✗ Error loading {os.path.basename(file)}: {e}")

if len(dfs) == 0:
    raise ValueError("❌ KHÔNG THỂ LOAD BẤT KỲ FILE NÀO! Kiểm tra lại đường dẫn hoặc RAM.")

# Merge tất cả
print(f"\n→ Đang merge {len(dfs)} files...")
df = pd.concat(dfs, ignore_index=True)
del dfs  # ← FIX: Giải phóng RAM ngay sau khi merge

print("\n" + "-" * 100)
print(f"→ Tổng số mẫu: {len(df):,}")
print(f"→ Số features: {len(df.columns)}")
print(f"→ Kích thước dataset: {df.shape}")

# ================================================================================
# 2. DATA ANALYSIS AND STATISTICS
# ================================================================================

print("\n" + "=" * 100)
print("STEP 2: PHÂN TÍCH VÀ THỐNG KÊ DATASET")
print("=" * 100)

# Tìm cột label (thường là cột cuối)
label_col = df.columns[-1]
print(f"\nCột nhãn: {label_col}")

# Thống kê nhãn gốc
print("\n" + "-" * 100)
print("PHÂN BỐ NHÃN GỐC:")
print("-" * 100)
label_counts = df[label_col].value_counts()
print(f"\n{'Tên nhãn':<50s} {'Số lượng':>15s} {'Tỷ lệ (%)':>10s}")
print("-" * 100)
for label, count in label_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{str(label):<50s} {count:>15,} {percentage:>10.2f}%")

# Chuyển đổi thành Binary labels
print("\n" + "-" * 100)
print("CHUYỂN ĐỔI THÀNH 2 LỚP: BENIGN vs ATTACK")
print("-" * 100)

def map_to_binary(label):
    label_lower = str(label).lower()
    if 'benign' in label_lower or 'normal' in label_lower:
        return 'Benign'
    else:
        return 'Attack'

df['binary_label'] = df[label_col].apply(map_to_binary)

# Thống kê Binary labels
print("\nPHÂN BỐ SAU KHI GỘP:")
print("-" * 100)
binary_counts = df['binary_label'].value_counts()
print(f"\n{'Nhãn':<15s} {'Số lượng':>15s} {'Tỷ lệ (%)':>10s}")
print("-" * 100)
for label, count in binary_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{label:<15s} {count:>15,} {percentage:>10.2f}%")

# Tính tỉ lệ mất cân bằng
imbalance_ratio = binary_counts.max() / binary_counts.min()
print(f"\n→ Tỉ lệ mất cân bằng (Imbalance Ratio): {imbalance_ratio:.2f}:1")

# Visualization
print("\n" + "-" * 100)
print("TẠO BIỂU ĐỒ PHÂN BỐ NHÃN")
print("-" * 100)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Pie chart
colors = ['#2ecc71', '#e74c3c']
explode = (0.05, 0)
axes[0].pie(binary_counts.values, labels=binary_counts.index, autopct='%1.2f%%',
           colors=colors, startangle=90, explode=explode, shadow=True)
axes[0].set_title('Phân bố nhãn - Pie Chart', fontsize=16, fontweight='bold', pad=20)

# Bar chart
bars = axes[1].bar(binary_counts.index, binary_counts.values, color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('Label', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Số lượng mẫu', fontsize=14, fontweight='bold')
axes[1].set_title('Phân bố nhãn - Bar Chart', fontsize=16, fontweight='bold', pad=20)
axes[1].grid(axis='y', alpha=0.3)

# Thêm giá trị lên cột
for bar, (label, count) in zip(bars, binary_counts.items()):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(BACKUP_FOLDER, 'label_distribution.png'), dpi=300, bbox_inches='tight')
print(f"✓ Đã lưu biểu đồ: {BACKUP_FOLDER}/label_distribution.png")
plt.close()  # ← FIX: Đóng figure thay vì show() (tránh crash trên WSL)

# ================================================================================
# 3. DATA PREPROCESSING
# ================================================================================

print("\n" + "=" * 100)
print("STEP 3: TIỀN XỬ LÝ DỮ LIỆU")
print("=" * 100)

# Tách features và labels
X = df.drop([label_col, 'binary_label'], axis=1)
y = df['binary_label']

print(f"\nShape ban đầu:")
print(f"  X: {X.shape}")
print(f"  y: {y.shape}")

# Xử lý missing values
print(f"\n→ Missing values: {X.isnull().sum().sum()}")
if X.isnull().sum().sum() > 0:
    print("  Filling missing values with 0...")
    X = X.fillna(0)

# Xử lý infinite values
print(f"→ Infinite values: {np.isinf(X.values).sum()}")
if np.isinf(X.values).sum() > 0:
    print("  Replacing infinite values with 0...")
    X = X.replace([np.inf, -np.inf], 0)

# Chuyển tất cả về numeric
print("→ Converting all columns to numeric...")
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(0)

# Loại bỏ constant columns
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    print(f"→ Loại bỏ {len(constant_cols)} constant columns")
    X = X.drop(constant_cols, axis=1)

print(f"\nShape sau xử lý:")
print(f"  X: {X.shape}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

print(f"\nLabel mapping: {label_mapping}")
print(f"  {le.classes_[0]} = {label_mapping[le.classes_[0]]}")
print(f"  {le.classes_[1]} = {label_mapping[le.classes_[1]]}")

# ================================================================================
# 4. SPLIT DATA
# ================================================================================

print("\n" + "=" * 100)
print("STEP 4: CHIA DỮ LIỆU")
print("=" * 100)

TEST_SIZE = 0.2
VAL_SIZE = 0.125  # 10% of total = 0.125 of train_val

# Split train+val and test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=42, stratify=y_encoded
)

# Split train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=VAL_SIZE, random_state=42, stratify=y_train_val
)

print(f"\nTổng số mẫu: {len(X):,}")
print(f"  Training:   {len(X_train):>8,} ({len(X_train)/len(X)*100:>5.1f}%)")
print(f"  Validation: {len(X_val):>8,} ({len(X_val)/len(X)*100:>5.1f}%)")
print(f"  Test:       {len(X_test):>8,} ({len(X_test)/len(X)*100:>5.1f}%)")

# Kiểm tra phân bố labels trong mỗi tập
print("\nPhân bố labels trong từng tập:")
print("-" * 100)
# ⚠️ FIX: Dùng thứ tự đúng theo LabelEncoder (Attack=0, Benign=1)
print(f"{'Tập':<15s} {le.classes_[0]:>12s} {le.classes_[1]:>12s}")
print("-" * 100)

for name, y_set in [('Training', y_train), ('Validation', y_val), ('Test', y_test)]:
    unique, counts = np.unique(y_set, return_counts=True)
    # Class 0 là Attack, Class 1 là Benign (theo alphabet)
    class0_count = counts[0] if unique[0] == 0 else counts[1]
    class1_count = counts[1] if unique[0] == 0 else counts[0]
    print(f"{name:<15s} {class0_count:>12,} {class1_count:>12,}")

# ================================================================================
# 5. DATA NORMALIZATION
# ================================================================================

print("\n" + "=" * 100)
print("STEP 5: CHUẨN HÓA DỮ LIỆU")
print("=" * 100)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Đã chuẩn hóa dữ liệu bằng StandardScaler")
print(f"  Mean: {scaler.mean_[:5]}... (first 5 features)")
print(f"  Std:  {scaler.scale_[:5]}... (first 5 features)")

# ================================================================================
# 6. BUILD CNN-GRU MODEL
# ================================================================================

print("\n" + "=" * 100)
print("STEP 6: XÂY DỰNG MÔ HÌNH CNN-GRU")
print("=" * 100)

def build_cnn_gru_model(input_shape, num_classes=2):
    """
    Xây dựng mô hình CNN-GRU theo kiến trúc DeepFed:
    - CNN Module: 3 Conv blocks
    - GRU Module: 2 GRU layers  
    - MLP Module: 2 Dense layers
    - Softmax output
    """
    
    input_layer = layers.Input(shape=input_shape, name='input')
    
    # ===== CNN MODULE =====
    print("\n→ Building CNN Module...")
    x_cnn = layers.Reshape((input_shape[0], 1), name='reshape_cnn')(input_layer)
    
    # Conv Block 1
    x_cnn = layers.Conv1D(filters=64, kernel_size=3, padding='same', 
                          activation='relu', name='conv1')(x_cnn)
    x_cnn = layers.BatchNormalization(name='bn1')(x_cnn)
    x_cnn = layers.MaxPooling1D(pool_size=2, name='pool1')(x_cnn)
    
    # Conv Block 2
    x_cnn = layers.Conv1D(filters=128, kernel_size=3, padding='same',
                          activation='relu', name='conv2')(x_cnn)
    x_cnn = layers.BatchNormalization(name='bn2')(x_cnn)
    x_cnn = layers.MaxPooling1D(pool_size=2, name='pool2')(x_cnn)
    
    # Conv Block 3
    x_cnn = layers.Conv1D(filters=256, kernel_size=3, padding='same',
                          activation='relu', name='conv3')(x_cnn)
    x_cnn = layers.BatchNormalization(name='bn3')(x_cnn)
    x_cnn = layers.MaxPooling1D(pool_size=2, name='pool3')(x_cnn)
    
    # Flatten
    cnn_output = layers.Flatten(name='flatten_cnn')(x_cnn)
    
    # ===== GRU MODULE =====
    print("→ Building GRU Module...")
    x_gru = layers.Reshape((input_shape[0], 1), name='reshape_gru')(input_layer)
    
    # GRU Layer 1
    x_gru = layers.GRU(units=128, return_sequences=True, name='gru1')(x_gru)
    
    # GRU Layer 2
    x_gru = layers.GRU(units=64, return_sequences=False, name='gru2')(x_gru)
    
    gru_output = x_gru
    
    # ===== CONCATENATE =====
    print("→ Concatenating CNN and GRU outputs...")
    concatenated = layers.Concatenate(name='concatenate')([cnn_output, gru_output])
    
    # ===== MLP MODULE =====
    print("→ Building MLP Module...")
    # Dense Layer 1
    x = layers.Dense(256, activation='relu', name='dense1')(concatenated)
    x = layers.BatchNormalization(name='bn_mlp1')(x)
    x = layers.Dropout(0.5, name='dropout1')(x)
    
    # Dense Layer 2
    x = layers.Dense(128, activation='relu', name='dense2')(x)
    x = layers.BatchNormalization(name='bn_mlp2')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    
    # ===== OUTPUT =====
    output = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = models.Model(inputs=input_layer, outputs=output, name='CNN_GRU_Model')
    
    return model

# Build model
input_shape = (X_train_scaled.shape[1],)
model = build_cnn_gru_model(input_shape, num_classes=2)

# Compile
print("\n→ Compiling model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✓ Đã xây dựng và compile model")
print(f"  Tổng số parameters: {model.count_params():,}")

# Model summary
print("\n" + "=" * 100)
print("KIẾN TRÚC MÔ HÌNH")
print("=" * 100)
model.summary()

# ================================================================================
# 7. TRAIN MODEL
# ================================================================================

print("\n" + "=" * 100)
print("STEP 7: HUẤN LUYỆN MÔ HÌNH")
print("=" * 100)

# Với 20 files (~14-15M samples), 20 epochs là đủ để model hội tụ
EPOCHS = 20  # Giảm xuống 20 epochs (đủ cho dataset lớn)
BATCH_SIZE = 2048  # Tăng batch size tối đa để tận dụng GPU 8GB

print(f"\nHyperparameters:")
print(f"  Epochs: {EPOCHS} (20 epochs đủ cho 20 files data)")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Optimizer: Adam (lr=0.001)")
print(f"  Loss: Sparse Categorical Crossentropy")

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,  # Paper DeepFed: patience 10-15 epochs
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,  # Giảm learning rate sau 5 epochs không cải thiện
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(BACKUP_FOLDER, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    # Note: Checkpoint tự động sẽ lưu best model, không cần lưu mỗi 5 epochs nữa
]

# ================================================================================
# CALCULATE CLASS WEIGHTS (để xử lý imbalanced data)
# ================================================================================

from sklearn.utils.class_weight import compute_class_weight

print("\n" + "-" * 100)
print("TÍNH TOÁN CLASS WEIGHTS")
print("-" * 100)

# Tính class weights tự động
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights_array))

print(f"\n→ Class Weights (để xử lý imbalanced data - ratio {imbalance_ratio:.2f}:1):")
print(f"   Class 0 ({le.classes_[0]}): {class_weights[0]:.6f}")
print(f"   Class 1 ({le.classes_[1]}): {class_weights[1]:.6f}")
print(f"\n→ Benign class được tăng trọng số {class_weights[1]/class_weights[0]:.2f}x so với Attack")
print(f"   → Model sẽ chú ý nhiều hơn vào minority class (Benign)")
print(f"   → Cải thiện Recall và F1-score cho class Benign")

print("\nBắt đầu training...\n")
print("⏰ Thời gian dự kiến: ~2-3 giờ cho 20 epochs với 20 files")
print("💾 Model sẽ tự động lưu:")
print(f"   - {BACKUP_FOLDER}/best_model.h5: Lưu model tốt nhất")
print("\n" + "=" * 100 + "\n")

# Training with error handling
try:
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,  # ← THÊM CLASS WEIGHTS để xử lý imbalanced data
        verbose=1
    )
    print("\n✓ Hoàn thành training!")
    
except KeyboardInterrupt:
    print("\n\n⚠️  TRAINING BỊ DỪNG BỞI NGƯỜI DÙNG (Ctrl+C)")
    print(f"→ Model tốt nhất đã được lưu tại: {BACKUP_FOLDER}/best_model.h5")
    print("→ Checkpoint cuối cùng có thể xem trong folder")
    raise
    
except Exception as e:
    print(f"\n\n❌ LỖI TRONG QUÁ TRÌNH TRAINING: {e}")
    print("→ Kiểm tra lại GPU memory hoặc RAM")
    print(f"→ Model tốt nhất trước khi lỗi đã được lưu tại: {BACKUP_FOLDER}/best_model.h5")
    raise

print("\n✓ Hoàn thành training!")

# ================================================================================
# 8. PLOT TRAINING HISTORY
# ================================================================================

print("\n" + "=" * 100)
print("STEP 8: VISUALIZE QUÁ TRÌNH TRAINING")
print("=" * 100)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Loss
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2, marker='o')
axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2, marker='s')
axes[0].set_title('Model Loss', fontsize=16, fontweight='bold', pad=15)
axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history.history['accuracy'], label='Train Acc', linewidth=2, marker='o')
axes[1].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2, marker='s')
axes[1].set_title('Model Accuracy', fontsize=16, fontweight='bold', pad=15)
axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BACKUP_FOLDER, 'training_history.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Đã lưu biểu đồ training history: {BACKUP_FOLDER}/training_history.png")
plt.close()  # ← FIX: Đóng figure thay vì show()

# ================================================================================
# 9. EVALUATE MODEL
# ================================================================================

print("\n" + "=" * 100)
print("STEP 9: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
print("=" * 100)

# Predictions
print("\n→ Đang thực hiện predictions trên tập test...")
y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

# ⚠️ QUAN TRỌNG: pos_label=0 vì Attack (class 0) là minority class quan trọng cần detect
# Class 0 = Attack (2.3%), Class 1 = Benign (97.7%)
# Trong Intrusion Detection, Attack là "positive" class (cần phát hiện)
precision = precision_score(y_test, y_pred, average='binary', pos_label=0)
recall = recall_score(y_test, y_pred, average='binary', pos_label=0)
f1 = f1_score(y_test, y_pred, average='binary', pos_label=0)

print("\n" + "=" * 100)
print("KẾT QUẢ ĐÁNH GIÁ")
print("=" * 100)
print(f"\n⚠️  Lưu ý: Metrics dưới đây tính cho class {le.classes_[0]} (pos_label=0)")
print(f"  → {le.classes_[0]} là minority class quan trọng cần phát hiện\n")
print(f"  ACCURACY:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  PRECISION: {precision:.4f} ({precision*100:.2f}%)  ← Precision của {le.classes_[0]}")
print(f"  RECALL:    {recall:.4f} ({recall*100:.2f}%)  ← Recall của {le.classes_[0]}")
print(f"  F1-SCORE:  {f1:.4f} ({f1*100:.2f}%)  ← F1-Score của {le.classes_[0]}")

# Classification Report
print("\n" + "-" * 100)
print("CLASSIFICATION REPORT CHI TIẾT")
print("-" * 100)

# ⚠️ QUAN TRỌNG: LabelEncoder tự động sort theo alphabet
# Attack (A) → 0, Benign (B) → 1
# Nên target_names phải theo đúng thứ tự này!
print(f"\n⚠️  Lưu ý: Label encoding:")
print(f"   Class 0 = {le.classes_[0]}")
print(f"   Class 1 = {le.classes_[1]}\n")

report = classification_report(y_test, y_pred, 
                               target_names=[le.classes_[0], le.classes_[1]],  # ← FIX: Dùng thứ tự đúng
                               digits=4)
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n" + "-" * 100)
print("CONFUSION MATRIX")
print("-" * 100)
print(f"\n⚠️  Lưu ý: Class 0 = {le.classes_[0]}, Class 1 = {le.classes_[1]}")
print(f"\n  True Negatives  (TN): {tn:>10,}  ← {le.classes_[0]} dự đoán đúng là {le.classes_[0]}")
print(f"  False Positives (FP): {fp:>10,}  ← {le.classes_[0]} nhầm thành {le.classes_[1]}")
print(f"  False Negatives (FN): {fn:>10,}  ← {le.classes_[1]} nhầm thành {le.classes_[0]}")
print(f"  True Positives  (TP): {tp:>10,}  ← {le.classes_[1]} dự đoán đúng là {le.classes_[1]}")

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))

# ⚠️ FIX: Dùng thứ tự label đúng theo LabelEncoder
# Attack (class 0) trước, Benign (class 1) sau
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=[le.classes_[0], le.classes_[1]],  # ← FIX: Attack, Benign
           yticklabels=[le.classes_[0], le.classes_[1]],  # ← FIX: Attack, Benign
           cbar_kws={'label': 'Count'},
           annot_kws={'fontsize': 14, 'fontweight': 'bold'})
plt.title('Confusion Matrix', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BACKUP_FOLDER, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Đã lưu confusion matrix: {BACKUP_FOLDER}/confusion_matrix.png")
plt.close()  # ← FIX: Đóng figure thay vì show()

# ================================================================================
# 10. SAVE RESULTS
# ================================================================================

print("\n" + "=" * 100)
print("STEP 10: LƯU KẾT QUẢ")
print("=" * 100)

# Save model
model.save(os.path.join(BACKUP_FOLDER, 'final_cnn_gru_model.h5'))
print(f"\n✓ Đã lưu model: {BACKUP_FOLDER}/final_cnn_gru_model.h5")

# Save scaler and label encoder
joblib.dump(scaler, os.path.join(BACKUP_FOLDER, 'scaler.pkl'))
print(f"✓ Đã lưu scaler: {BACKUP_FOLDER}/scaler.pkl")

joblib.dump(le, os.path.join(BACKUP_FOLDER, 'label_encoder.pkl'))
print(f"✓ Đã lưu label encoder: {BACKUP_FOLDER}/label_encoder.pkl")

# Save detailed results
with open(os.path.join(BACKUP_FOLDER, 'results_summary.txt'), 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write(" " * 30 + "KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH CNN-GRU\n")
    f.write("=" * 100 + "\n\n")
    
    f.write("DATASET INFORMATION\n")
    f.write("-" * 100 + "\n")
    f.write(f"Dataset: IoT Dataset 2023\n")
    f.write(f"Total samples: {len(df):,}\n")
    f.write(f"Number of features: {X_train_scaled.shape[1]}\n")
    f.write(f"Number of classes: 2 (Benign, Attack)\n\n")
    
    f.write("DATA SPLIT\n")
    f.write("-" * 100 + "\n")
    f.write(f"Training:   {len(X_train):>10,} ({len(X_train)/len(X)*100:>5.1f}%)\n")
    f.write(f"Validation: {len(X_val):>10,} ({len(X_val)/len(X)*100:>5.1f}%)\n")
    f.write(f"Test:       {len(X_test):>10,} ({len(X_test)/len(X)*100:>5.1f}%)\n\n")
    
    f.write("MODEL ARCHITECTURE\n")
    f.write("-" * 100 + "\n")
    f.write(f"Model: CNN-GRU (DeepFed Architecture)\n")
    f.write(f"Total parameters: {model.count_params():,}\n\n")
    
    f.write("TRAINING CONFIGURATION\n")
    f.write("-" * 100 + "\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Optimizer: Adam (lr=0.001)\n")
    f.write(f"Loss function: Sparse Categorical Crossentropy\n\n")
    
    f.write("PERFORMANCE METRICS\n")
    f.write("-" * 100 + "\n")
    f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Precision: {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"Recall:    {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)\n\n")
    
    f.write("CONFUSION MATRIX\n")
    f.write("-" * 100 + "\n")
    f.write(f"True Negatives  (TN): {tn:,}\n")
    f.write(f"False Positives (FP): {fp:,}\n")
    f.write(f"False Negatives (FN): {fn:,}\n")
    f.write(f"True Positives  (TP): {tp:,}\n\n")
    
    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 100 + "\n")
    f.write(report)

print(f"✓ Đã lưu kết quả chi tiết: {BACKUP_FOLDER}/results_summary.txt")

# Save training config info
with open(os.path.join(BACKUP_FOLDER, 'training_config.txt'), 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("TRAINING CONFIGURATION\n")
    f.write("=" * 100 + "\n\n")
    f.write(f"Timestamp: {TIMESTAMP}\n")
    f.write(f"Backup Folder: {BACKUP_FOLDER}\n\n")
    f.write(f"Dataset Files: 20 files\n")
    f.write(f"Total Samples: {len(df):,}\n")
    f.write(f"Features: {X_train_scaled.shape[1]}\n\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Learning Rate: 0.001\n")
    f.write(f"Optimizer: Adam\n")
    f.write(f"Class Weights: Yes (balanced)\n")
    f.write(f"  - Class 0 ({le.classes_[0]}): {class_weights[0]:.6f}\n")
    f.write(f"  - Class 1 ({le.classes_[1]}): {class_weights[1]:.6f}\n")
    f.write(f"  - Weight ratio: {class_weights[1]/class_weights[0]:.2f}x\n")

print(f"✓ Đã lưu cấu hình training: {BACKUP_FOLDER}/training_config.txt")

# ================================================================================
# FINAL SUMMARY
# ================================================================================

print("\n" + "=" * 100)
print(" " * 40 + "HOÀN THÀNH!")
print("=" * 100)

print("\n📊 KẾT QUẢ CUỐI CÙNG:")
print("-" * 100)
print(f"  ✓ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  ✓ Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  ✓ Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"  ✓ F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

print("\n📁 CÁC FILE ĐÃ LƯU:")
print("-" * 100)
print(f"  📂 Thư mục backup: {BACKUP_FOLDER}/")
print(f"  ✓ final_cnn_gru_model.h5      - Model đã train")
print(f"  ✓ best_model.h5               - Model tốt nhất (từ checkpoint)")
print(f"  ✓ scaler.pkl                  - StandardScaler")
print(f"  ✓ label_encoder.pkl           - LabelEncoder")
print(f"  ✓ results_summary.txt         - Kết quả chi tiết")
print(f"  ✓ training_config.txt         - Cấu hình training")
print(f"  ✓ label_distribution.png      - Biểu đồ phân bố nhãn")
print(f"  ✓ training_history.png        - Quá trình training")
print(f"  ✓ confusion_matrix.png        - Confusion matrix")

print("\n" + "=" * 100)
print(" " * 35 + "Cảm ơn bạn đã sử dụng!")
print("=" * 100 + "\n")