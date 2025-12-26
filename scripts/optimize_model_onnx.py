"""
Optimize YOLOv8 Model using ONNX with INT8 Quantization
This script converts your PyTorch model to optimized ONNX format
"""

import os
from pathlib import Path
from ultralytics import YOLO
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import shutil

def optimize_yolov8_to_onnx(
    model_path='models/best.pt',
    output_dir='models',
    img_size=640
):
    """
    Optimize YOLOv8 model to ONNX with INT8 quantization
    
    Args:
        model_path: Path to your YOLOv8 .pt model
        output_dir: Directory to save optimized models
        img_size: Input image size (default 640)
    """
    
    print("=" * 60)
    print("⚡ YOLOV8 TO ONNX OPTIMIZATION PIPELINE")
    print("=" * 60)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Define output paths
    onnx_fp32_path = Path(output_dir) / 'best_fp32.onnx'
    onnx_int8_path = Path(output_dir) / 'best_int8.onnx'
    
    # ========================================
    # STEP 1: Load YOLOv8 Model
    # ========================================
    print("\n1️⃣ Loading YOLOv8 Model...")
    
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found at {model_path}")
        return None
    
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # ========================================
    # STEP 2: Export to ONNX (FP32 - Full Precision)
    # ========================================
    print("\n2️⃣ Exporting to ONNX (FP32 - Full Precision)...")
    print("   This creates the baseline ONNX model...")
    
    try:
        # YOLOv8 has built-in ONNX export
        model.export(
            format='onnx',
            imgsz=img_size,
            simplify=True,  # Simplify the model
            opset=12,       # ONNX opset version
        )
        
        # YOLOv8 saves as best.onnx by default, let's rename it
        default_onnx_path = Path(model_path).with_suffix('.onnx')
        
        if default_onnx_path.exists():
            # Only rename if not already named correctly
            if default_onnx_path != onnx_fp32_path:
                shutil.move(str(default_onnx_path), str(onnx_fp32_path))
            print(f"✅ FP32 ONNX model saved to: {onnx_fp32_path}")
        else:
            print(f"❌ Error: Expected ONNX file not found at {default_onnx_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error exporting to ONNX: {e}")
        return None
    
    # Check FP32 model size
    fp32_size_mb = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
    print(f"   FP32 Model Size: {fp32_size_mb:.2f} MB")
    
    # ========================================
    # STEP 3: Apply INT8 Quantization
    # ========================================
    print("\n3️⃣ Applying INT8 Quantization...")
    print("   Converting FP32 (32-bit floats) → INT8 (8-bit integers)...")
    print("   This reduces model size by ~75%!")
    
    try:
        # Dynamic quantization (weights only)
        # This is the easiest and most compatible method
        quantize_dynamic(
            model_input=str(onnx_fp32_path),
            model_output=str(onnx_int8_path),
            weight_type=QuantType.QUInt8  # Quantize to 8-bit unsigned integers
        )
        
        print(f"✅ INT8 ONNX model saved to: {onnx_int8_path}")
        
    except Exception as e:
        print(f"❌ Error during quantization: {e}")
        print("   Tip: Try installing onnxruntime with: pip install onnxruntime")
        return None
    
    # Check INT8 model size
    int8_size_mb = os.path.getsize(onnx_int8_path) / (1024 * 1024)
    print(f"   INT8 Model Size: {int8_size_mb:.2f} MB")
    
    # ========================================
    # STEP 4: Calculate Improvements
    # ========================================
    print("\n4️⃣ Calculating Optimization Results...")
    
    # Original PyTorch model size
    pt_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    # Size reduction (PT to INT8)
    size_reduction = ((pt_size_mb - int8_size_mb) / pt_size_mb) * 100
    
    print("\n" + "=" * 60)
    print("📊 OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Original PyTorch (.pt):  {pt_size_mb:.2f} MB")
    print(f"ONNX FP32:               {fp32_size_mb:.2f} MB")
    print(f"ONNX INT8 (Optimized):   {int8_size_mb:.2f} MB")
    print(f"Size Reduction:          {size_reduction:.1f}%")
    print("=" * 60)
    
    # ========================================
    # STEP 5: Validate Models
    # ========================================
    print("\n5️⃣ Validating ONNX Models...")
    
    try:
        # Validate FP32
        onnx_model_fp32 = onnx.load(str(onnx_fp32_path))
        onnx.checker.check_model(onnx_model_fp32)
        print("✅ FP32 model is valid")
        
        # Validate INT8
        onnx_model_int8 = onnx.load(str(onnx_int8_path))
        onnx.checker.check_model(onnx_model_int8)
        print("✅ INT8 model is valid")
        
    except Exception as e:
        print(f"⚠️ Warning: Model validation failed: {e}")
    
    # ========================================
    # STEP 6: Save Optimization Report
    # ========================================
    print("\n6️⃣ Saving Optimization Report...")
    
    with open('optimization_report.txt', 'w') as f:
        f.write("MODEL OPTIMIZATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("OPTIMIZATION METHOD: ONNX INT8 Dynamic Quantization\n\n")
        f.write(f"Original Model (.pt):     {pt_size_mb:.2f} MB\n")
        f.write(f"ONNX FP32:                {fp32_size_mb:.2f} MB\n")
        f.write(f"ONNX INT8 (Optimized):    {int8_size_mb:.2f} MB\n")
        f.write(f"Size Reduction:           {size_reduction:.1f}%\n\n")
        f.write("=" * 60 + "\n\n")
        f.write("FILES CREATED:\n")
        f.write(f"  - {onnx_fp32_path}\n")
        f.write(f"  - {onnx_int8_path}\n")
    
    print("✅ Optimization report saved to 'optimization_report.txt'")
    
    print("\n✨ OPTIMIZATION COMPLETE! ✨")
    print(f"\nYour optimized model is ready: {onnx_int8_path}")
    print("\nNext step: Run 'measure_optimized_metrics.py' to measure performance!")
    
    return {
        'pt_size_mb': pt_size_mb,
        'fp32_size_mb': fp32_size_mb,
        'int8_size_mb': int8_size_mb,
        'size_reduction': size_reduction,
        'int8_model_path': str(onnx_int8_path)
    }


if __name__ == "__main__":
    # Run optimization
    results = optimize_yolov8_to_onnx(
        model_path='models/best.pt',
        output_dir='models',
        img_size=640
    )