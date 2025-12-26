"""
Measure Baseline Model Metrics
Run this script to measure your current YOLOv8 model's performance
"""

import os
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2

def measure_baseline_metrics(model_path='models/best.pt', test_image_path=None):
    """
    Measure baseline model metrics: size and inference time
    
    Args:
        model_path: Path to your YOLOv8 model
        test_image_path: Path to test image (optional, creates dummy if None)
    """
    
    print("=" * 60)
    print("📊 BASELINE MODEL METRICS MEASUREMENT")
    print("=" * 60)
    
    # 1. MODEL SIZE
    print("\n1️⃣ Measuring Model Size...")
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found at {model_path}")
        return None
    
    model_size_bytes = os.path.getsize(model_path)
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    print(f"✅ Model Path: {model_path}")
    print(f"✅ Model Size: {model_size_mb:.2f} MB ({model_size_bytes:,} bytes)")
    
    # 2. LOAD MODEL
    print("\n2️⃣ Loading Model...")
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # 3. PREPARE TEST IMAGE
    print("\n3️⃣ Preparing Test Image...")
    if test_image_path and Path(test_image_path).exists():
        test_img = cv2.imread(test_image_path)
        print(f"✅ Using provided image: {test_image_path}")
    else:
        # Create a dummy 640x640 image for testing
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        print("✅ Using dummy 640x640 image")
    
    # 4. MEASURE INFERENCE TIME (Multiple runs for accuracy)
    print("\n4️⃣ Measuring Inference Time...")
    print("Running 20 inference tests (excluding first warmup run)...")
    
    inference_times = []
    
    # Warmup run (not counted)
    _ = model.predict(test_img, verbose=False)
    
    # Actual measurement runs
    for i in range(20):
        start_time = time.time()
        results = model.predict(test_img, conf=0.25, verbose=False)
        end_time = time.time()
        
        inference_time_ms = (end_time - start_time) * 1000
        inference_times.append(inference_time_ms)
        
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/20 runs...")
    
    # Calculate statistics
    avg_inference_time = np.mean(inference_times)
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)
    std_inference_time = np.std(inference_times)
    
    print("\n✅ Inference Time Measurement Complete")
    print(f"   Average: {avg_inference_time:.2f} ms")
    print(f"   Min: {min_inference_time:.2f} ms")
    print(f"   Max: {max_inference_time:.2f} ms")
    print(f"   Std Dev: {std_inference_time:.2f} ms")
    
    # 5. CALCULATE FPS
    avg_fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
    
    # 6. SUMMARY
    print("\n" + "=" * 60)
    print("📋 BASELINE MODEL SUMMARY")
    print("=" * 60)
    print(f"Model Size:        {model_size_mb:.2f} MB")
    print(f"Inference Time:    {avg_inference_time:.2f} ms (average)")
    print(f"FPS:               {avg_fps:.1f}")
    print("=" * 60)
    
    # 7. SAVE RESULTS TO FILE
    results_dict = {
        'model_path': model_path,
        'model_size_mb': model_size_mb,
        'model_size_bytes': model_size_bytes,
        'avg_inference_time_ms': avg_inference_time,
        'min_inference_time_ms': min_inference_time,
        'max_inference_time_ms': max_inference_time,
        'std_inference_time_ms': std_inference_time,
        'avg_fps': avg_fps
    }
    
    # Save to text file
    with open('reports/baseline_metrics.txt', 'w') as f:
        f.write("BASELINE MODEL METRICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model Size: {model_size_mb:.2f} MB\n")
        f.write(f"Inference Time (Avg): {avg_inference_time:.2f} ms\n")
        f.write(f"Inference Time (Min): {min_inference_time:.2f} ms\n")
        f.write(f"Inference Time (Max): {max_inference_time:.2f} ms\n")
        f.write(f"FPS: {avg_fps:.1f}\n")
        f.write("=" * 60 + "\n")
    
    print("\n💾 Results saved to 'baseline_metrics.txt'")
    
    return results_dict


if __name__ == "__main__":
    # Run the measurement
    # Option 1: Use with default dummy image
    # results = measure_baseline_metrics('models/best.pt')
    
    # Option 2: Use with your own test image (uncomment below)
    results = measure_baseline_metrics('models/best.pt', 'assets/test_image_pump.jpg')