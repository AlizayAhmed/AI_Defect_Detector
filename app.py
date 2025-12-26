"""
AI Defect Detector - Edge Optimized
Main Streamlit Application with 3 Tabs: Baseline, Optimized, Comparison
Uses Ultralytics YOLO for consistent postprocessing across all models
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time
import json
from pathlib import Path
import warnings
from fpdf import FPDF
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Page config
st.set_page_config(
    page_title="AI Defect Detector - Edge Optimized",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        font-size: 1rem;
        padding: 0.75rem;
        border-radius: 10px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_directories():
    """Ensure required directories exist"""
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

@st.cache_resource
def load_baseline_model():
    """Load baseline ONNX FP32 model using Ultralytics"""
    try:
        model_path = 'models/best_fp32.onnx'
        if Path(model_path).exists():
            model = YOLO(model_path)
            return model
        return None
    except Exception as e:
        st.error(f"Error loading baseline model: {str(e)}")
        return None

@st.cache_resource
def load_optimized_model():
    """Load optimized ONNX INT8 model using Ultralytics"""
    try:
        model_path = 'models/best_int8.onnx'
        if Path(model_path).exists():
            model = YOLO(model_path)
            return model
        return None
    except Exception as e:
        st.error(f"Error loading optimized model: {str(e)}")
        return None

def process_with_yolo(image, model, conf_threshold):
    """Process image with YOLO model (with warmup)"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    
    # Warmup run
    _ = model.predict(img_array, conf=conf_threshold, iou=0.7, verbose=False)
    
    # Measured run
    start_time = time.time()
    results = model.predict(img_array, conf=conf_threshold, iou=0.7, verbose=False)
    inference_time = (time.time() - start_time) * 1000
    
    # Get annotated image
    annotated_img = results[0].plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Extract detections
    detections = []
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            detection = {
                'class': results[0].names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xyxy[0].tolist()
            }
            detections.append(detection)
    
    return annotated_img, detections, inference_time

def save_result(result_type, data):
    """Save detection result to JSON"""
    file_path = Path(f"results/{result_type}_result.json")
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_result(result_type):
    """Load detection result from JSON"""
    file_path = Path(f"results/{result_type}_result.json")
    if file_path.exists():
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def clear_result(result_type):
    """Clear detection result"""
    file_path = Path(f"results/{result_type}_result.json")
    if file_path.exists():
        file_path.unlink()

def get_current_image(tab_name):
    """Get current image for a tab (tab-specific or shared)"""
    tab_image_key = f'{tab_name}_image'
    if tab_image_key in st.session_state and st.session_state[tab_image_key] is not None:
        return st.session_state[tab_image_key]
    return st.session_state.get('shared_image', None)

def generate_pdf_report(baseline_data, optimized_data):
    """Generate PDF comparison report"""
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 10, 'AI Defect Detection - Comparison Report', 0, 1, 'C')
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
    pdf.ln(10)
    
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Performance Metrics Comparison', 0, 1)
    pdf.ln(2)
    
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(60, 8, 'Metric', 1, 0, 'C')
    pdf.cell(40, 8, 'Baseline', 1, 0, 'C')
    pdf.cell(40, 8, 'Optimized', 1, 0, 'C')
    pdf.cell(40, 8, 'Improvement', 1, 1, 'C')
    
    pdf.set_font('Arial', '', 10)
    
    time_diff = ((baseline_data['inference_time'] - optimized_data['inference_time']) / baseline_data['inference_time']) * 100
    pdf.cell(60, 8, 'Inference Time (ms)', 1, 0)
    pdf.cell(40, 8, f"{baseline_data['inference_time']:.2f}", 1, 0, 'C')
    pdf.cell(40, 8, f"{optimized_data['inference_time']:.2f}", 1, 0, 'C')
    pdf.cell(40, 8, f"{time_diff:.1f}% faster", 1, 1, 'C')
    
    pdf.cell(60, 8, 'Defects Detected', 1, 0)
    pdf.cell(40, 8, str(baseline_data['num_detections']), 1, 0, 'C')
    pdf.cell(40, 8, str(optimized_data['num_detections']), 1, 0, 'C')
    pdf.cell(40, 8, 'Same' if baseline_data['num_detections'] == optimized_data['num_detections'] else 'Different', 1, 1, 'C')
    
    baseline_fps = 1000 / baseline_data['inference_time']
    optimized_fps = 1000 / optimized_data['inference_time']
    pdf.cell(60, 8, 'FPS', 1, 0)
    pdf.cell(40, 8, f"{baseline_fps:.1f}", 1, 0, 'C')
    pdf.cell(40, 8, f"{optimized_fps:.1f}", 1, 0, 'C')
    pdf.cell(40, 8, f"+{((optimized_fps - baseline_fps) / baseline_fps * 100):.1f}%", 1, 1, 'C')
    
    pdf.ln(10)
    
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Detection Details', 0, 1)
    pdf.ln(2)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, f'Baseline Model (ONNX FP32) - {baseline_data["num_detections"]} defects found:', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    for i, det in enumerate(baseline_data['detections'][:5], 1):
        pdf.cell(0, 6, f"{i}. {det['class'].upper()} - Confidence: {det['confidence']:.1%}", 0, 1)
    
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, f'Optimized Model (ONNX INT8) - {optimized_data["num_detections"]} defects found:', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    for i, det in enumerate(optimized_data['detections'][:5], 1):
        pdf.cell(0, 6, f"{i}. {det['class'].upper()} - Confidence: {det['confidence']:.1%}", 0, 1)
    
    pdf.ln(10)
    
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Model Information', 0, 1)
    pdf.ln(2)
    
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, 
        'Baseline Model: YOLOv8n (ONNX FP32) - 11.70 MB, Float32 precision\n'
        'Optimized Model: YOLOv8n (ONNX INT8) - 3.20 MB, INT8 quantization\n'
        'Size Reduction: 72.6% | Suitable for edge deployment on Raspberry Pi\n'
        'Postprocessing: Ultralytics YOLO (consistent across all models)'
    )
    
    pdf_path = f"reports/comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(pdf_path)
    
    return pdf_path

# ============================================================
# MAIN APP
# ============================================================

def main():
    ensure_directories()
    
    st.markdown('<p class="main-header">🔍 AI Defect Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Edge-Optimized: ONNX FP32 vs INT8 with Consistent Detection</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("---")
        
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence for detection"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info("""
        **Baseline:** ONNX FP32 (11.7 MB)  
        **Optimized:** ONNX INT8 (3.2 MB)  
        **Dataset:** NEU Surface Defects  
        **Classes:** 6 defect types  
        **Engine:** Ultralytics YOLO  
        **Settings:** conf=0.25, iou=0.7
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ Defect Types")
        st.markdown("""
        - 🔴 Crazing
        - 🟠 Inclusion
        - 🟡 Patches
        - 🟢 Pitted Surface
        - 🔵 Rolled-in Scale
        - 🟣 Scratches
        """)
    
    st.markdown("### 📤 Upload Shared Image")
    
    if 'shared_image' not in st.session_state:
        st.session_state.shared_image = None
    
    if 'shared_uploader_key' not in st.session_state:
        st.session_state.shared_uploader_key = 0
    
    uploaded_file = st.file_uploader(
        "Choose an image (used by all models by default)...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image - both models will use this unless you change it per tab",
        key=f"shared_uploader_{st.session_state.shared_uploader_key}"
    )
    
    if uploaded_file:
        st.session_state.shared_image = Image.open(uploaded_file)
        st.session_state.shared_filename = uploaded_file.name
        st.success(f"✅ Shared image uploaded: {uploaded_file.name}")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🔵 Baseline Model (FP32)", "🟢 Optimized Model (INT8)", "📊 Comparison"])
    
    # ============================================================
    # TAB 1: BASELINE MODEL
    # ============================================================
    with tab1:
        st.markdown("## 🔵 Baseline Model (ONNX FP32)")
        
        current_image = get_current_image('baseline')
        
        if not current_image:
            st.info("👆 Please upload a shared image first")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### 📷 Current Image")
                
                # Show which image is being used
                if 'baseline_image' in st.session_state and st.session_state.baseline_image is not None:
                    st.caption("🔹 Using tab-specific image")
                else:
                    st.caption("🔷 Using shared image")
                
                st.image(current_image, use_container_width=True)
                
                # Change Image button with its own uploader
                if 'baseline_uploader_key' not in st.session_state:
                    st.session_state.baseline_uploader_key = 0
                
                change_file = st.file_uploader(
                    "Upload different image for baseline",
                    type=['jpg', 'jpeg', 'png'],
                    key=f"baseline_change_{st.session_state.baseline_uploader_key}",
                    help="Upload a different image just for this tab"
                )
                
                if change_file:
                    st.session_state.baseline_image = Image.open(change_file)
                    st.toast("✅ Baseline image changed!", icon="🔄")
                    time.sleep(0.3)
                    st.rerun()
            
            with col2:
                st.markdown("### 🔍 Detection")
                
                baseline_model = load_baseline_model()
                
                if not baseline_model:
                    st.error("❌ Baseline model (best_fp32.onnx) not found!")
                    st.stop()
                
                btn_col1, btn_col2 = st.columns(2)
                
                with btn_col1:
                    if st.button("🔍 Detect Defects", key="detect_baseline", type="primary", use_container_width=True):
                        with st.spinner("🔄 Analyzing with baseline model..."):
                            try:
                                annotated_img, detections, inference_time = process_with_yolo(
                                    current_image,
                                    baseline_model,
                                    conf_threshold
                                )
                                
                                result_data = {
                                    'inference_time': inference_time,
                                    'num_detections': len(detections),
                                    'detections': detections,
                                    'timestamp': datetime.now().isoformat()
                                }
                                save_result('baseline', result_data)
                                
                                st.session_state.baseline_annotated = annotated_img
                                st.session_state.baseline_data = result_data
                                
                                st.toast("✅ Detection complete!", icon="✅")
                                time.sleep(0.5)
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                
                with btn_col2:
                    if st.button("🗑️ Clear Results", key="clear_baseline", use_container_width=True):
                        # Clear detection results
                        clear_result('baseline')
                        if 'baseline_annotated' in st.session_state:
                            del st.session_state.baseline_annotated
                        if 'baseline_data' in st.session_state:
                            del st.session_state.baseline_data
                        
                        # Clear tab-specific image (revert to shared)
                        if 'baseline_image' in st.session_state:
                            st.session_state.baseline_image = None
                        st.session_state.baseline_uploader_key += 1
                        
                        st.toast("✅ Results cleared! Reverted to shared image.", icon="🗑️")
                        time.sleep(0.5)
                        st.rerun()
            
            baseline_data = load_result('baseline')
            
            if baseline_data:
                st.markdown("---")
                st.markdown("### 📊 Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Inference Time", f"{baseline_data['inference_time']:.2f} ms")
                with col2:
                    st.metric("🎯 Defects Found", baseline_data['num_detections'])
                with col3:
                    fps = 1000 / baseline_data['inference_time']
                    st.metric("📹 FPS", f"{fps:.1f}")
                
                if 'baseline_annotated' in st.session_state:
                    st.image(st.session_state.baseline_annotated, caption="Detected Defects", width=600)
                
                if baseline_data['detections']:
                    st.markdown("### 🔍 Detection Details")
                    sorted_detections = sorted(baseline_data['detections'], key=lambda x: x['confidence'], reverse=True)
                    
                    for idx, det in enumerate(sorted_detections, 1):
                        with st.expander(f"**{idx}. {det['class'].upper()}** - Confidence: {det['confidence']:.1%}"):
                            conf_percent = det['confidence'] * 100
                            
                            if det['confidence'] > 0.8:
                                color = "🟢"
                                label = "High Confidence"
                            elif det['confidence'] > 0.5:
                                color = "🟡"
                                label = "Medium Confidence"
                            else:
                                color = "🟠"
                                label = "Low Confidence"
                            
                            st.write(f"{color} **{label}**")
                            st.progress(det['confidence'])
                            st.write(f"**Confidence:** {conf_percent:.1f}%")
                            st.write(f"**Defect Type:** {det['class'].title()}")
                else:
                    st.success("✅ No defects detected!")
    
    # ============================================================
    # TAB 2: OPTIMIZED MODEL
    # ============================================================
    with tab2:
        st.markdown("## 🟢 Optimized Model (ONNX INT8)")
        
        current_image = get_current_image('optimized')
        
        if not current_image:
            st.info("👆 Please upload a shared image first")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### 📷 Current Image")
                
                # Show which image is being used
                if 'optimized_image' in st.session_state and st.session_state.optimized_image is not None:
                    st.caption("🔹 Using tab-specific image")
                else:
                    st.caption("🔷 Using shared image")
                
                st.image(current_image, use_container_width=True)
                
                # Change Image button with its own uploader
                if 'optimized_uploader_key' not in st.session_state:
                    st.session_state.optimized_uploader_key = 0
                
                change_file = st.file_uploader(
                    "Upload different image for optimized",
                    type=['jpg', 'jpeg', 'png'],
                    key=f"optimized_change_{st.session_state.optimized_uploader_key}",
                    help="Upload a different image just for this tab"
                )
                
                if change_file:
                    st.session_state.optimized_image = Image.open(change_file)
                    st.toast("✅ Optimized image changed!", icon="🔄")
                    time.sleep(0.3)
                    st.rerun()
            
            with col2:
                st.markdown("### 🔍 Detection")
                
                optimized_model = load_optimized_model()
                
                if not optimized_model:
                    st.error("❌ Optimized model (best_int8.onnx) not found!")
                    st.stop()
                
                btn_col1, btn_col2 = st.columns(2)
                
                with btn_col1:
                    if st.button("🔍 Detect Defects", key="detect_optimized", type="primary", use_container_width=True):
                        with st.spinner("🔄 Analyzing with optimized model..."):
                            try:
                                annotated_img, detections, inference_time = process_with_yolo(
                                    current_image,
                                    optimized_model,
                                    conf_threshold
                                )
                                
                                result_data = {
                                    'inference_time': inference_time,
                                    'num_detections': len(detections),
                                    'detections': detections,
                                    'timestamp': datetime.now().isoformat()
                                }
                                save_result('optimized', result_data)
                                
                                st.session_state.optimized_annotated = annotated_img
                                st.session_state.optimized_data = result_data
                                
                                st.toast("✅ Detection complete!", icon="✅")
                                time.sleep(0.5)
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                
                with btn_col2:
                    if st.button("🗑️ Clear Results", key="clear_optimized", use_container_width=True):
                        # Clear detection results
                        clear_result('optimized')
                        if 'optimized_annotated' in st.session_state:
                            del st.session_state.optimized_annotated
                        if 'optimized_data' in st.session_state:
                            del st.session_state.optimized_data
                        
                        # Clear tab-specific image (revert to shared)
                        if 'optimized_image' in st.session_state:
                            st.session_state.optimized_image = None
                        st.session_state.optimized_uploader_key += 1
                        
                        st.toast("✅ Results cleared! Reverted to shared image.", icon="🗑️")
                        time.sleep(0.5)
                        st.rerun()
            
            optimized_data = load_result('optimized')
            
            if optimized_data:
                st.markdown("---")
                st.markdown("### 📊 Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Inference Time", f"{optimized_data['inference_time']:.2f} ms")
                with col2:
                    st.metric("🎯 Defects Found", optimized_data['num_detections'])
                with col3:
                    fps = 1000 / optimized_data['inference_time']
                    st.metric("📹 FPS", f"{fps:.1f}")
                
                if 'optimized_annotated' in st.session_state:
                    st.image(st.session_state.optimized_annotated, caption="Detected Defects", width=600)
                
                if optimized_data['detections']:
                    st.markdown("### 🔍 Detection Details")
                    sorted_detections = sorted(optimized_data['detections'], key=lambda x: x['confidence'], reverse=True)
                    
                    for idx, det in enumerate(sorted_detections, 1):
                        with st.expander(f"**{idx}. {det['class'].upper()}** - Confidence: {det['confidence']:.1%}"):
                            conf_percent = det['confidence'] * 100
                            
                            if det['confidence'] > 0.8:
                                color = "🟢"
                                label = "High Confidence"
                            elif det['confidence'] > 0.5:
                                color = "🟡"
                                label = "Medium Confidence"
                            else:
                                color = "🟠"
                                label = "Low Confidence"
                            
                            st.write(f"{color} **{label}**")
                            st.progress(det['confidence'])
                            st.write(f"**Confidence:** {conf_percent:.1f}%")
                            st.write(f"**Defect Type:** {det['class'].title()}")
                else:
                    st.success("✅ No defects detected!")
    
    # ============================================================
    # TAB 3: COMPARISON
    # ============================================================
    with tab3:
        st.markdown("## 📊 Model Comparison")
        
        baseline_data = load_result('baseline')
        optimized_data = load_result('optimized')
        
        if not baseline_data or not optimized_data:
            st.warning("⚠️ Please run detection on BOTH models first!")
            
            missing = []
            if not baseline_data:
                missing.append("Baseline Model (FP32)")
            if not optimized_data:
                missing.append("Optimized Model (INT8)")
            
            st.info(f"Missing results from: {', '.join(missing)}")
            st.stop()
        
        st.markdown("### 📷 Detection Results Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔵 Baseline (ONNX FP32)")
            if 'baseline_annotated' in st.session_state:
                st.image(st.session_state.baseline_annotated, use_container_width=True)
            else:
                st.info("Run baseline detection to see results")
        
        with col2:
            st.markdown("#### 🟢 Optimized (ONNX INT8)")
            if 'optimized_annotated' in st.session_state:
                st.image(st.session_state.optimized_annotated, use_container_width=True)
            else:
                st.info("Run optimized detection to see results")
        
        st.markdown("---")
        
        st.markdown("### 📈 Performance Metrics Comparison")
        
        time_diff = ((baseline_data['inference_time'] - optimized_data['inference_time']) / baseline_data['inference_time']) * 100
        baseline_fps = 1000 / baseline_data['inference_time']
        optimized_fps = 1000 / optimized_data['inference_time']
        fps_diff = ((optimized_fps - baseline_fps) / baseline_fps) * 100
        
        comparison_data = {
            'Metric': ['Model Size (MB)', 'Inference Time (ms)', 'Defects Detected', 'FPS'],
            'Baseline (FP32)': [
                '11.70',
                f"{baseline_data['inference_time']:.2f}",
                str(baseline_data['num_detections']),
                f"{baseline_fps:.1f}"
            ],
            'Optimized (INT8)': [
                '3.20',
                f"{optimized_data['inference_time']:.2f}",
                str(optimized_data['num_detections']),
                f"{optimized_fps:.1f}"
            ],
            'Improvement': [
                '72.6% smaller',
                f"{time_diff:.1f}% faster" if time_diff > 0 else f"{abs(time_diff):.1f}% slower",
                "Same ✅" if baseline_data['num_detections'] == optimized_data['num_detections'] else "Different ⚠️",
                f"+{fps_diff:.1f}%" if fps_diff > 0 else f"{fps_diff:.1f}%"
            ]
        }
        
        st.table(comparison_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Speed Improvement",
                f"{time_diff:.1f}%",
                delta=f"{abs(baseline_data['inference_time'] - optimized_data['inference_time']):.2f} ms",
                delta_color="inverse"
            )
        with col2:
            st.metric(
                "FPS Gain",
                f"{fps_diff:.1f}%",
                delta=f"{abs(optimized_fps - baseline_fps):.1f} FPS"
            )
        with col3:
            detection_match = baseline_data['num_detections'] == optimized_data['num_detections']
            st.metric(
                "Detection Accuracy",
                "Matched ✅" if detection_match else "Different ⚠️",
                delta=None
            )
        
        st.markdown("---")
        
        st.markdown("### 🔍 Detection Details Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔵 Baseline Detections")
            if baseline_data['detections']:
                for idx, det in enumerate(sorted(baseline_data['detections'], key=lambda x: x['confidence'], reverse=True), 1):
                    st.write(f"**{idx}. {det['class'].upper()}** - {det['confidence']:.1%}")
            else:
                st.write("No defects detected")
        
        with col2:
            st.markdown("#### 🟢 Optimized Detections")
            if optimized_data['detections']:
                for idx, det in enumerate(sorted(optimized_data['detections'], key=lambda x: x['confidence'], reverse=True), 1):
                    st.write(f"**{idx}. {det['class'].upper()}** - {det['confidence']:.1%}")
            else:
                st.write("No defects detected")
        
        st.markdown("---")
        
        st.markdown("### 📥 Export Report")
        
        if st.button("📥 Download PDF Report", type="primary", use_container_width=True):
            with st.spinner("🔄 Generating PDF report..."):
                try:
                    pdf_path = generate_pdf_report(baseline_data, optimized_data)
                    
                    with open(pdf_path, 'rb') as f:
                        st.download_button(
                            label="📄 Download PDF",
                            data=f.read(),
                            file_name=Path(pdf_path).name,
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    st.success(f"✅ PDF report generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p style='margin: 0;'>🤖 YOLOv8n | ⚡ ONNX FP32 vs INT8 | 🎯 Ultralytics Postprocessing</p>
            <p style='margin: 0.5rem 0 0 0;'>📚 AI-Based Surface Defect Detection System</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()