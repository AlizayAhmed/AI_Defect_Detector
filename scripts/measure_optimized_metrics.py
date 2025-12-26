"""
Measure Optimized ONNX Model Metrics
Run this after optimization to measure INT8 model performance
"""

import os
import time
import numpy as np
from pathlib import Path
import onnxruntime as ort
import cv2

def measure_optimized_metrics(
    onnx_model_path='models/best_int8.onnx',
    test_image_path=None,
    img_size=640
):
    """
    Measure optimized ONNX model metrics: size and inference time
    
    Args:
        onnx_model_path: Path to your ONNX INT8 model
        test_image_path: Path to test image (optional)
        img_size: Input image size (default 640)
    """
    
    print("=" * 60)
    print("📊 OPTIMIZED MODEL METRICS MEASUREMENT")
    print("=" * 60)
    
    # 1. MODEL SIZE
    print("\n1️⃣ Measuring Model Size...")
    if not Path(onnx_model_path).exists():
        print(f"❌ Error: Model not found at {onnx_model_path}")
        return None
    
    model_size_bytes = os.path.getsize(onnx_model_path)
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    print(f"✅ Model Path: {onnx_model_path}")
    print(f"✅ Model Size: {model_size_mb:.2f} MB ({model_size_bytes:,} bytes)")
    
    # 2. LOAD ONNX MODEL
    print("\n2️⃣ Loading ONNX Model...")
    try:
        # Create ONNX Runtime session
        session = ort.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider']  # Use CPU (edge deployment scenario)
        )
        print("✅ ONNX model loaded successfully")
        
        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"   Input name: {input_name}")
        print(f"   Output names: {output_names}")
        
    except Exception as e:
        print(f"❌ Error loading ONNX model: {e}")
        print("   Tip: Install onnxruntime with: pip install onnxruntime")
        return None
    
    # 3. PREPARE TEST IMAGE
    print("\n3️⃣ Preparing Test Image...")
    if test_image_path and Path(test_image_path).exists():
        img = cv2.imread(test_image_path)
        print(f"✅ Using provided image: {test_image_path}")
    else:
        # Create a dummy image
        img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        print(f"✅ Using dummy {img_size}x{img_size} image")
    
    # Preprocess image for ONNX (YOLOv8 format)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize and transpose: HWC -> CHW
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC to CHW
    
    # Add batch dimension: CHW -> NCHW
    input_tensor = np.expand_dims(img_transposed, axis=0)
    
    print(f"✅ Input tensor shape: {input_tensor.shape}")
    
    # 4. MEASURE INFERENCE TIME
    print("\n4️⃣ Measuring Inference Time...")
    print("Running 20 inference tests (excluding first warmup run)...")
    
    inference_times = []
    
    # Warmup run (not counted)
    _ = session.run(output_names, {input_name: input_tensor})
    
    # Actual measurement runs
    for i in range(20):
        start_time = time.time()
        outputs = session.run(output_names, {input_name: input_tensor})
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
    print("📋 OPTIMIZED MODEL SUMMARY")
    print("=" * 60)
    print(f"Model Size:        {model_size_mb:.2f} MB")
    print(f"Inference Time:    {avg_inference_time:.2f} ms (average)")
    print(f"FPS:               {avg_fps:.1f}")
    print("=" * 60)
    
    # 7. SAVE RESULTS TO FILE
    results_dict = {
        'model_path': onnx_model_path,
        'model_size_mb': model_size_mb,
        'model_size_bytes': model_size_bytes,
        'avg_inference_time_ms': avg_inference_time,
        'min_inference_time_ms': min_inference_time,
        'max_inference_time_ms': max_inference_time,
        'std_inference_time_ms': std_inference_time,
        'avg_fps': avg_fps
    }
    
    # Save to text file
    with open('reports/optimized_metrics.txt', 'w') as f:
        f.write("OPTIMIZED MODEL METRICS (ONNX INT8)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model Size: {model_size_mb:.2f} MB\n")
        f.write(f"Inference Time (Avg): {avg_inference_time:.2f} ms\n")
        f.write(f"Inference Time (Min): {min_inference_time:.2f} ms\n")
        f.write(f"Inference Time (Max): {max_inference_time:.2f} ms\n")
        f.write(f"FPS: {avg_fps:.1f}\n")
        f.write("=" * 60 + "\n")
    
    print("\n💾 Results saved to 'optimized_metrics.txt'")
    
    return results_dict


def compare_models(baseline_metrics_file='baseline_metrics.txt',
                   optimized_metrics_file='optimized_metrics.txt'):
    """
    Create a comparison table between baseline and optimized models
    """
    
    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON TABLE")
    print("=" * 60)
    
    # Read baseline metrics
    baseline_data = {}
    if Path(baseline_metrics_file).exists():
        with open(baseline_metrics_file, 'r') as f:
            for line in f:
                if 'Model Size:' in line:
                    baseline_data['size'] = float(line.split(':')[1].strip().split()[0])
                elif 'Inference Time (Avg):' in line:
                    baseline_data['time'] = float(line.split(':')[1].strip().split()[0])
    
    # Read optimized metrics
    optimized_data = {}
    if Path(optimized_metrics_file).exists():
        with open(optimized_metrics_file, 'r') as f:
            for line in f:
                if 'Model Size:' in line:
                    optimized_data['size'] = float(line.split(':')[1].strip().split()[0])
                elif 'Inference Time (Avg):' in line:
                    optimized_data['time'] = float(line.split(':')[1].strip().split()[0])
    
    if not baseline_data or not optimized_data:
        print("⚠️ Could not find both metrics files. Run both measurement scripts first.")
        return
    
    # Calculate improvements
    size_reduction = ((baseline_data['size'] - optimized_data['size']) / baseline_data['size']) * 100
    time_improvement = ((baseline_data['time'] - optimized_data['time']) / baseline_data['time']) * 100
    
    # Print comparison table
    print("\n┌─────────────────────────┬──────────────┬──────────────┬─────────────┐")
    print("│ Metric                  │ Baseline     │ Optimized    │ Improvement │")
    print("├─────────────────────────┼──────────────┼──────────────┼─────────────┤")
    print(f"│ Model Size (MB)         │ {baseline_data['size']:>12.2f} │ {optimized_data['size']:>12.2f} │ {size_reduction:>10.1f}% │")
    print(f"│ Inference Time (ms)     │ {baseline_data['time']:>12.2f} │ {optimized_data['time']:>12.2f} │ {time_improvement:>10.1f}% │")
    print("└─────────────────────────┴──────────────┴──────────────┴─────────────┘")
    
    # Save comparison
    with open('reports/comparison_table.txt', 'w') as f:
        f.write("MODEL COMPARISON: BASELINE vs OPTIMIZED\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Metric':<25} {'Baseline':<15} {'Optimized':<15} {'Improvement'}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model Size (MB)':<25} {baseline_data['size']:<15.2f} {optimized_data['size']:<15.2f} {size_reduction:.1f}%\n")
        f.write(f"{'Inference Time (ms)':<25} {baseline_data['time']:<15.2f} {optimized_data['time']:<15.2f} {time_improvement:.1f}%\n")
        f.write("=" * 70 + "\n")
    
    print("\n💾 Comparison saved to 'comparison_table.txt'")

if __name__ == "__main__":
    # Measure optimized model
    results = measure_optimized_metrics('models/best_int8.onnx', 'assets/test_image_pump.jpg')
    
    # If both metrics files exist, create comparison
    if Path('baseline_metrics.txt').exists():
        compare_models()