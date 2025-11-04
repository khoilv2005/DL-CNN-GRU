import tensorflow as tf

print("=" * 100)
print(" " * 35 + "KIỂM TRA GPU/CPU")
print("=" * 100)

# Kiểm tra TensorFlow version
print(f"\nTensorFlow version: {tf.__version__}")

# Kiểm tra GPU có available không
print("\n" + "-" * 100)
print("DANH SÁCH DEVICES:")
print("-" * 100)

physical_devices = tf.config.list_physical_devices()
for device in physical_devices:
    print(f"  {device.device_type}: {device.name}")

# Kiểm tra GPU cụ thể
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("\n" + "-" * 100)
    print(f"✓ TÌM THẤY {len(gpus)} GPU!")
    print("-" * 100)
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
        # Lấy thông tin chi tiết
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            print(f"    Device name: {gpu_details.get('device_name', 'N/A')}")
        except:
            pass
else:
    print("\n" + "-" * 100)
    print("✗ KHÔNG TÌM THẤY GPU - Đang dùng CPU")
    print("-" * 100)
    print("\nĐể sử dụng GPU, bạn cần:")
    print("  1. NVIDIA GPU (GTX/RTX series)")
    print("  2. CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)")
    print("  3. cuDNN (https://developer.nvidia.com/cudnn)")
    print("  4. TensorFlow-GPU: pip install tensorflow[and-cuda]")

# Test simple computation
print("\n" + "-" * 100)
print("TEST COMPUTATION:")
print("-" * 100)

import time

# Test trên device hiện tại
with tf.device('/CPU:0'):
    start_time = time.time()
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
    cpu_time = time.time() - start_time
    print(f"  CPU time: {cpu_time:.4f} seconds")

if gpus:
    with tf.device('/GPU:0'):
        start_time = time.time()
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        gpu_time = time.time() - start_time
        print(f"  GPU time: {gpu_time:.4f} seconds")
        print(f"  Speedup: {cpu_time/gpu_time:.2f}x faster")

print("\n" + "=" * 100)
print("KẾT LUẬN:")
print("=" * 100)
if gpus:
    print("✓ Model SẼ TRAIN TRÊN GPU (nhanh hơn 5-10 lần)")
else:
    print("✗ Model SẼ TRAIN TRÊN CPU (chậm hơn)")
    print("\nƯỚC TÍNH THỜI GIAN:")
    print("  - Với CPU: 2-4 giờ (dataset lớn)")
    print("  - Với GPU: 20-40 phút")
print("=" * 100)
