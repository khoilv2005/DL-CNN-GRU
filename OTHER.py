# Buoc 1: Import các thư viện cần thiết
import numpy as np
import tensorflow as tf
import pandas as pd
import glob
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, GRU, Dense, Dropout, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

# Buoc 2: Định nghĩa kiến trúc mô hình CNN-GRU (Giữ nguyên)
def build_cnn_gru_model(input_shape, num_classes=2):
    # Input layer
    inputs = Input(shape=input_shape)

    # CNN Module (3 convolutional blocks)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=256, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    cnn_out = Flatten()(x)

    # GRU Module (2 GRU layers)
    # Reshape for GRU
    x_reshaped = tf.keras.layers.Reshape((-1, 1))(inputs)
    gru_out = GRU(128, return_sequences=True)(x_reshaped)
    gru_out = GRU(64)(gru_out)

    # Concatenate CNN and GRU outputs
    combined = Concatenate()([cnn_out, gru_out])

    # MLP Module (2 fully connected layers + dropout)
    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Buoc 3: Tải và Xử lý dữ liệu (Giữ nguyên phần xử lý, chỉ thay đổi bước chia dữ liệu cuối cùng)
# ... (Giữ nguyên toàn bộ phần code tải và xử lý dữ liệu của bạn từ câu trả lời trước) ...
# Bắt đầu từ đây
# Đường dẫn đến thư mục chứa các file pcap.csv
dataset_path = './IoT_Dataset_2023/'

# Tìm tất cả các file .csv
all_files = glob.glob(os.path.join(dataset_path, '**/*.csv'), recursive=True)

if not all_files:
    print(f"LỖI: Không tìm thấy file .csv nào trong đường dẫn '{dataset_path}'")
else:
    print(f"Tìm thấy {len(all_files)} file CSV. Bắt đầu xử lý theo từng phần để tiết kiệm RAM...")

    # ===================================================================
    # BƯỚC MỚI: XỬ LÝ DỮ LIỆU THEO TỪNG PHẦN (CHUNKING)
    # ===================================================================
    
    # Bước 1: Xác định các cột số cần thiết bằng cách đọc một file mẫu
    try:
        sample_df = pd.read_csv(all_files[0], nrows=10) # Chỉ đọc 10 dòng đầu
        all_columns = sample_df.columns.tolist()
        numeric_columns = sample_df.select_dtypes(include=np.number).columns.tolist()
        print(f"Tìm thấy {len(numeric_columns)} cột số và {len(all_columns)} tổng số cột.")
    except Exception as e:
        print(f"Lỗi khi đọc file mẫu: {e}")
        raise

    # Bước 2: Đọc và xử lý dữ liệu theo từng chunk
    chunk_size = 50000  # Kích thước mỗi khúc, có thể tăng/giảm tùy RAM
    processed_chunks = []
    
    for file_path in all_files:
        try:
            print(f"Đang xử lý file: {os.path.basename(file_path)}")

            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunk.dropna(axis=1, how='all', inplace=True)
                chunk.dropna(inplace=True)
                
                available_cols = [col for col in all_columns if col in chunk.columns]
                chunk = chunk[available_cols]
                
                for col in numeric_columns:
                    if col in chunk.columns:
                        if chunk[col].dtype == 'float64':
                            chunk[col] = chunk[col].astype('float32')
                        elif chunk[col].dtype == 'int64':
                            chunk[col] = chunk[col].astype('int32')
                
                if not chunk.empty:
                    processed_chunks.append(chunk)

        except Exception as e:
            print(f"Lỗi khi xử lý file {file_path}: {e}")
    
    gc.collect()

    # Bước 3: Ghép tất cả các chunk đã xử lý lại với nhau
    if not processed_chunks:
        print("\nLỖI: Không có dữ liệu nào được xử lý thành công.")
    else:
        print(f"\nĐã xử lý xong tất cả các file. Bắt đầu ghép {len(processed_chunks)} khúc dữ liệu...")
        data = pd.concat(processed_chunks, ignore_index=True)
        
        del processed_chunks
        gc.collect()
        
        print(f"Tổng số dòng sau khi ghép: {len(data)}")
        print("Phân phối nhãn TRƯỚC khi xử lý:")
        print(data['Label'].value_counts())

        # ===================================================================
        # BƯỚC MỚI: GOM TẤT CẢ CÁC TẤN CÔNG THÀNH "Attack"
        # ===================================================================
        print("\nĐang gom tất cả các loại tấn công thành nhãn 'Attack'...")
        data['Label'] = data['Label'].apply(lambda x: 'Benign' if str(x).lower() == 'benign' else 'Attack')
        
        print("Phân phối nhãn SAU khi xử lý:")
        print(data['Label'].value_counts())

        # ===================================================================
        # BƯỚC MỚI: XỬ LÝ CÁC GIÁ TRỊ VÔ HẠN (INFINITY)
        # ===================================================================
        print("\nĐang xử lý các giá trị vô hạn (inf) để tránh lỗi...")
        numeric_cols_for_cleaning = data.select_dtypes(include=np.number).columns
        
        data[numeric_cols_for_cleaning] = data[numeric_cols_for_cleaning].replace([np.inf, -np.inf], np.nan)
        
        if data[numeric_cols_for_cleaning].isnull().any().any():
            print("Phát hiện các giá trị NaN sau khi thay thế inf. Đang điền giá trị trung bình...")
            data.fillna(data.mean(), inplace=True)
        
        print("Đã xử lý xong các giá trị vô hạn.")

        # ===================================================================
        # TIẾP TỤC CÁC BƯỚC XỬ LÝ DỮ LIỆU NHƯ BÌNH THƯỜNG
        # ===================================================================

        # 4. Chọn đặc trưng (nếu có nhiều đặc trưng)
        feature_cols = [col for col in data.columns if col != 'Label']
        if len(feature_cols) > 26:
            X = data[feature_cols]
            y = data['Label']
            
            selector = SelectKBest(f_classif, k=26)
            X_new = selector.fit_transform(X, y)
            
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
            print(f"26 đặc trưng được chọn: {selected_features}")
            
            data_selected = pd.DataFrame(X_new, columns=selected_features)
            data_selected['Label'] = y.values
        else:
            data_selected = data.copy()

        del data
        gc.collect()

        # 5. Chuẩn hóa dữ liệu
        feature_columns = [col for col in data_selected.columns if col != 'Label']
        X = data_selected[feature_columns].values
        y = data_selected['Label'].values

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y) 
        y_categorical = tf.keras.utils.to_categorical(y_encoded)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        print(f"\nKích thước dữ liệu cuối cùng: {X_reshaped.shape}")
        print(f"Số lượng lớp: {len(np.unique(y_encoded))} (0: Attack, 1: Benign)")

        # ===================================================================
        # BƯỚC MỚI: CHIA DỮ LIỆU CHO MÔ HÌNH TẬP TRUNG
        # ===================================================================
        # Thay vì chia cho các agent, chúng ta chia thành tập train và test
        X_train, X_test, y_train, y_test = train_test_split(
            X_reshaped, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
        )

        print(f"\nKích thước tập huấn luyện: {X_train.shape}")
        print(f"Kích thước tập kiểm tra: {X_test.shape}")

        # ===================================================================
        # BƯỚC MỚI: XÂY DỰNG, HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH TẬP TRUNG
        # ===================================================================
        
        # 1. Xây dựng mô hình
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = 2
        model = build_cnn_gru_model(input_shape, num_classes)
        model.summary()

        # 2. Biên dịch mô hình
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # 3. Huấn luyện mô hình
        print("\nBắt đầu huấn luyện mô hình...")
        epochs = 20 # Số vòng lặp huấn luyện
        batch_size = 64
        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_test, y_test),
                            verbose=1)

        # 4. Đánh giá mô hình trên tập kiểm tra
        print("\nĐánh giá mô hình trên tập kiểm tra...")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        # 5. Dự đoán và tạo báo cáo chi tiết
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)

        print("\nBáo cáo phân loại:")
        print(classification_report(y_true, y_pred, target_names=['Attack', 'Benign']))

        # 6. Vẽ ma trận nhầm lẫn
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Attack', 'Benign'], yticklabels=['Attack', 'Benign'])
        plt.xlabel('Dự đoán')
        plt.ylabel('Thực tế')
        plt.title('Ma trận nhầm lẫn')
        plt.show()

        # 7. Vẽ biểu đồ quá trình huấn luyện
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Biểu đồ Accuracy qua các Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Biểu đồ Loss qua các Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()