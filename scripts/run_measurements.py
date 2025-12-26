"""
Run all measurements with the SAME test image for fair comparison
"""

from measure_baseline_metrics import measure_baseline_metrics
from measure_optimized_metrics import measure_optimized_metrics, compare_models

# Use the same test image for both measurements
TEST_IMAGE = 'assets/test_image_pump.jpg'

print("🚀 Running complete measurement pipeline with consistent test image\n")

# Step 1: Measure Baseline
print("📊 STEP 1: Measuring Baseline Model")
print("=" * 60)
baseline_results = measure_baseline_metrics('models/best.pt', TEST_IMAGE)

if not baseline_results:
    print("❌ Baseline measurement failed!")
    exit(1)

print("\n" + "=" * 60)
input("Press Enter to continue to optimized measurement...")

# Step 2: Measure Optimized
print("\n📊 STEP 2: Measuring Optimized Model")
print("=" * 60)
optimized_results = measure_optimized_metrics('models/best_int8.onnx', TEST_IMAGE)

if not optimized_results:
    print("❌ Optimized measurement failed!")
    exit(1)

# Step 3: Compare
print("\n📊 STEP 3: Generating Comparison Table")
print("=" * 60)
compare_models()

print("\n✅ All measurements complete with consistent test image!")
print(f"   Test image used: {TEST_IMAGE}")