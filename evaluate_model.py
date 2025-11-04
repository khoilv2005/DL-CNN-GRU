import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 100)
print(" " * 25 + "ĐÁNH GIÁ MODEL CNN-GRU ĐÃ TRAIN XONG")
print("=" * 100)

# ================================================================================
# 1. LOAD DATASET (giống y như code training)
# ================================================================================

print("\n" + "=" * 100)
print("STEP 1: LOAD DATASET")
print("=" * 100)

DATA_PATH = './IoT_Dataset_2023'

csv_files = []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))

csv_files = sorted(csv_files)[:10]

print(f"\nLoading {len(csv_files)} CSV files...")
dfs = []
for file in csv_files:
    try:
        df_temp = pd.read_csv(file)
        dfs.append(df_temp)
        print(f"✓ {os.path.basename(file):50s} - {len(df_temp):>10,} samples")
    except Exception as e:
        print(f"✗ Error: {file}")

df = pd.concat(dfs, ignore_index=True)
print(f"\n→ Total: {len(df):,} samples, {len(df.columns)} features")

# ================================================================================
# 2. PREPROCESSING (giống y như code training)
# ================================================================================

print("\n" + "=" * 100)
print("STEP 2: PREPROCESSING")
print("=" * 100)

label_col = df.columns[-1]

def map_to_binary(label):
    label_lower = str(label).lower()
    if 'benign' in label_lower or 'normal' in label_lower:
        return 'Benign'
    else:
        return 'Attack'

df['binary_label'] = df[label_col].apply(map_to_binary)

X = df.drop([label_col, 'binary_label'], axis=1)
y = df['binary_label']

# Xử lý missing/infinite values
X = X.fillna(0)
X = X.replace([np.inf, -np.inf], 0)

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(0)

# Loại bỏ constant columns
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    X = X.drop(constant_cols, axis=1)

print(f"\n✓ Features shape: {X.shape}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"✓ Labels: {le.classes_[0]}={le.transform([le.classes_[0]])[0]}, {le.classes_[1]}={le.transform([le.classes_[1]])[0]}")

# ================================================================================
# 3. SPLIT DATA (giống y như code training)
# ================================================================================

print("\n" + "=" * 100)
print("STEP 3: SPLIT DATA")
print("=" * 100)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val
)

print(f"\n✓ Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✓ Data normalized")

# ================================================================================
# 4. LOAD MODEL ĐÃ TRAIN
# ================================================================================

print("\n" + "=" * 100)
print("STEP 4: LOAD MODEL ĐÃ TRAIN")
print("=" * 100)

if os.path.exists('best_model.h5'):
    print("\n→ Loading best_model.h5...")
    model = keras.models.load_model('best_model.h5')
    print("✓ Đã load model thành công!")
    print(f"  Total parameters: {model.count_params():,}")
else:
    print("\n✗ KHÔNG TÌM THẤY FILE best_model.h5!")
    print("  Bạn cần train model trước bằng cách chạy DL.py")
    exit(1)

# ================================================================================
# 5. EVALUATE ON VALIDATION SET
# ================================================================================

print("\n" + "=" * 100)
print("STEP 5: ĐÁNH GIÁ TRÊN VALIDATION SET")
print("=" * 100)

print("\n→ Predicting on validation set...")
y_val_pred_proba = model.predict(X_val_scaled, verbose=0)
y_val_pred = np.argmax(y_val_pred_proba, axis=1)

val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred, average='binary')
val_recall = recall_score(y_val, y_val_pred, average='binary')
val_f1 = f1_score(y_val, y_val_pred, average='binary')

print("\n" + "-" * 100)
print("VALIDATION SET RESULTS:")
print("-" * 100)
print(f"  Accuracy:  {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"  Precision: {val_precision:.4f} ({val_precision*100:.2f}%)")
print(f"  Recall:    {val_recall:.4f} ({val_recall*100:.2f}%)")
print(f"  F1-Score:  {val_f1:.4f} ({val_f1*100:.2f}%)")

# ================================================================================
# 6. EVALUATE ON TEST SET
# ================================================================================

print("\n" + "=" * 100)
print("STEP 6: ĐÁNH GIÁ TRÊN TEST SET")
print("=" * 100)

print("\n→ Predicting on test set...")
y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print("\n" + "=" * 100)
print("TEST SET RESULTS (KẾT QUẢ CHÍNH THỨC):")
print("=" * 100)
print(f"\n  ACCURACY:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  PRECISION: {precision:.4f} ({precision*100:.2f}%)")
print(f"  RECALL:    {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1-SCORE:  {f1:.4f} ({f1*100:.2f}%)")

# Classification Report
print("\n" + "-" * 100)
print("CLASSIFICATION REPORT:")
print("-" * 100)
report = classification_report(y_test, y_pred, 
                               target_names=['Benign', 'Attack'],
                               digits=4)
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n" + "-" * 100)
print("CONFUSION MATRIX:")
print("-" * 100)
print(f"\n  True Negatives  (TN): {tn:>10,}")
print(f"  False Positives (FP): {fp:>10,}")
print(f"  False Negatives (FN): {fn:>10,}")
print(f"  True Positives  (TP): {tp:>10,}")

# ================================================================================
# 7. VISUALIZATIONS
# ================================================================================

print("\n" + "=" * 100)
print("STEP 7: TẠO BIỂU ĐỒ")
print("=" * 100)

# Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Benign', 'Attack'],
           yticklabels=['Benign', 'Attack'],
           cbar_kws={'label': 'Count'},
           annot_kws={'fontsize': 14, 'fontweight': 'bold'})
plt.title('Confusion Matrix - Test Set', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_final.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: confusion_matrix_final.png")
plt.close()

# Metrics Comparison
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Metrics - Test Set', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

# Add values on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.4f}\n({value*100:.2f}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: metrics_comparison.png")
plt.close()

# ================================================================================
# 8. SAVE RESULTS
# ================================================================================

print("\n" + "=" * 100)
print("STEP 8: LƯU KẾT QUẢ")
print("=" * 100)

import joblib
joblib.dump(scaler, 'scaler_final.pkl')
joblib.dump(le, 'label_encoder_final.pkl')
print("\n✓ Saved: scaler_final.pkl, label_encoder_final.pkl")

# Save detailed results
with open('evaluation_results.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write(" " * 25 + "KẾT QUẢ ĐÁNH GIÁ MODEL CNN-GRU\n")
    f.write("=" * 100 + "\n\n")
    
    f.write("VALIDATION SET RESULTS:\n")
    f.write("-" * 100 + "\n")
    f.write(f"Accuracy:  {val_accuracy:.4f} ({val_accuracy*100:.2f}%)\n")
    f.write(f"Precision: {val_precision:.4f} ({val_precision*100:.2f}%)\n")
    f.write(f"Recall:    {val_recall:.4f} ({val_recall*100:.2f}%)\n")
    f.write(f"F1-Score:  {val_f1:.4f} ({val_f1*100:.2f}%)\n\n")
    
    f.write("TEST SET RESULTS (OFFICIAL):\n")
    f.write("-" * 100 + "\n")
    f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Precision: {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"Recall:    {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)\n\n")
    
    f.write("CONFUSION MATRIX:\n")
    f.write("-" * 100 + "\n")
    f.write(f"True Negatives  (TN): {tn:,}\n")
    f.write(f"False Positives (FP): {fp:,}\n")
    f.write(f"False Negatives (FN): {fn:,}\n")
    f.write(f"True Positives  (TP): {tp:,}\n\n")
    
    f.write("CLASSIFICATION REPORT:\n")
    f.write("-" * 100 + "\n")
    f.write(report)

print("✓ Saved: evaluation_results.txt")

# ================================================================================
# FINAL SUMMARY
# ================================================================================

print("\n" + "=" * 100)
print(" " * 40 + "HOÀN THÀNH!")
print("=" * 100)

print("\n📊 KẾT QUẢ CUỐI CÙNG (TEST SET):")
print("-" * 100)
print(f"  ✓ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  ✓ Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  ✓ Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"  ✓ F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

print("\n📁 FILES ĐÃ LƯU:")
print("-" * 100)
print("  ✓ confusion_matrix_final.png  - Confusion matrix")
print("  ✓ metrics_comparison.png      - So sánh các metrics")
print("  ✓ evaluation_results.txt      - Kết quả chi tiết")
print("  ✓ scaler_final.pkl            - StandardScaler")
print("  ✓ label_encoder_final.pkl     - LabelEncoder")

print("\n" + "=" * 100 + "\n")
