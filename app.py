import streamlit as st
import numpy as np
from PIL import Image
import os
import onnxruntime as ort
import requests
from requests.auth import HTTPDigestAuth
import io

# Set page configurations
st.set_page_config(
    page_title="🥦 Produce Freshness Predictor",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-end UI/UX aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main container adjustments */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header card design */
    .header-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        margin-bottom: 2.5rem;
        text-align: center;
    }
    .header-card h1 {
        color: white !important;
        font-weight: 700 !important;
        font-size: 2.6rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .header-card p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    /* Custom Card Style for details */
    .premium-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar styling overrides */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
        padding-top: 2rem;
    }
    
    /* Metrics panel decoration */
    .metric-container {
        background: #f1f5f9;
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.2s ease-in-out !important;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px -1px rgba(37, 99, 235, 0.3) !important;
    }
    
    /* Live/Demo indicators */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .status-active {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .status-simulated {
        background-color: #fef9c3;
        color: #854d0e;
        border: 1px solid #fef08a;
    }
</style>
""", unsafe_allow_html=True)

# Custom header
st.markdown("""
<div class="header-card">
    <h1>🥦 AI Freshness & Shelf-Life Predictor</h1>
    <p>Real-time Quality Control System for Quick-Commerce Warehouse Intake</p>
</div>
""", unsafe_allow_html=True)

# Cache ONNX model loading
@st.cache_resource
def load_model():
    if not os.path.exists('best_model_v2.onnx'):
        st.info("Downloading model... please wait (~30 seconds)")
        url = 'https://drive.google.com/uc?export=download&id=1ISydbI2W2oMrZlLShuiQHn6tmQVFCQKb'
        session = requests.Session()
        response = session.get(url, stream=True)

        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                url = url + '&confirm=' + value
                response = session.get(url, stream=True)
                break

        with open('best_model_v2.onnx', 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)

    return ort.InferenceSession('best_model_v2.onnx')

try:
    session = load_model()
except Exception as e:
    st.error(f"Error loading ONNX model: {e}")
    st.stop()

# Initialize session state for tracking current active image & metadata
if 'active_image' not in st.session_state:
    st.session_state.active_image = None
if 'source_info' not in st.session_state:
    st.session_state.source_info = None

# Sidebar for settings & model specifications
with st.sidebar:
    st.markdown("### 📊 System Specs")
    st.markdown("""
    - **Model Architecture:** MobileNetV2
    - **Transfer Learning:** ONNX Export
    - **Test Dataset:** 3,040 images
    - **Intake Categories:** Banana, Tomato, Potato
    """)
    st.markdown("---")
    
    # Active Image Details
    if st.session_state.active_image is not None:
        st.markdown("### 🔍 Active Inspection")
        st.info(f"**Source:** {st.session_state.source_info}")
        if st.button("🧹 Clear Inspection Data"):
            st.session_state.active_image = None
            st.session_state.source_info = None
            st.rerun()

# Layout: Main columns
col_inputs, col_results = st.columns([1, 1], gap="large")

with col_inputs:
    st.markdown("### 📥 Select Image Source")
    
    tab_upload, tab_camera = st.tabs([
        "📁 Upload Image File", 
        "📷 Axis Network Camera"
    ])
    
    # Tab 1: Uploading local image files
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload produce snapshot (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            key="file_uploader"
        )
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file).convert("RGB")
                st.session_state.active_image = img
                st.session_state.source_info = f"File Upload: {uploaded_file.name}"
            except Exception as e:
                st.error(f"Error loading uploaded file: {e}")
                
    # Tab 2: Axis Camera snapshot capture
    with tab_camera:
        # Toggle demo/simulation mode
        demo_mode = st.toggle("Enable Demo / Simulation Mode", value=True, help="Simulate Axis camera snapshot responses without active physical hardware")
        
        if demo_mode:
            st.markdown('<span class="status-badge status-simulated">Demo Mode Active</span>', unsafe_allow_html=True)
            
            # Select demo options
            mock_option = st.selectbox(
                "Select Simulated Camera Target",
                options=[
                    "Fresh Banana",
                    "Spoiled Banana",
                    "Fresh Tomato",
                    "Spoiled Tomato",
                    "Fresh Potato",
                    "Spoiled Potato"
                ]
            )
            
            if st.button("📸 Trigger Simulated Snapshot"):
                # Map option to file
                mapping = {
                    "Fresh Banana": "banana_fresh.png",
                    "Spoiled Banana": "banana_spoil.png",
                    "Fresh Tomato": "tomato_fresh.png",
                    "Spoiled Tomato": "tomato_spoil.png",
                    "Fresh Potato": "potato_fresh.png",
                    "Spoiled Potato": "potato_spoil.png"
                }
                filename = mapping[mock_option]
                mock_path = os.path.join("mock_images", filename)
                
                if os.path.exists(mock_path):
                    try:
                        img = Image.open(mock_path).convert("RGB")
                        st.session_state.active_image = img
                        st.session_state.source_info = f"Axis Camera Snapshot (Simulated: {mock_option})"
                        st.toast(f"Captured snapshot of {mock_option} from simulated camera", icon="✅")
                    except Exception as e:
                        st.error(f"Failed to open mock image: {e}")
                else:
                    st.error(f"Mock image file {mock_path} not found. Please verify folder contents.")
                    
        else:
            st.markdown('<span class="status-badge status-active">Live Axis VAPIX Integration</span>', unsafe_allow_html=True)
            
            # Axis camera configurations
            cam_ip = st.text_input("Camera Hostname / IP", value="192.168.68.106")
            
            col_proto, col_port = st.columns(2)
            with col_proto:
                cam_proto = st.selectbox("Protocol", ["http", "https"], index=1)
            with col_port:
                cam_port = st.number_input("Port", min_value=1, max_value=65535, value=80 if cam_proto == "http" else 443)
                
            col_chan, col_res = st.columns(2)
            with col_chan:
                cam_channel = st.number_input("Video Channel ID", min_value=1, max_value=64, value=1)
            with col_res:
                cam_res = st.selectbox("Snapshot Resolution", ["1920x1080", "1280x720", "800x600", "640x480", "320x240"], index=1)
                
            st.markdown("**Authentication Settings**")
            cam_user = st.text_input("Username", value="VLTuser")
            cam_pass = st.text_input("Password", type="password", value="XXXXXXXXXXXXXXXXX")
            cam_auth = st.selectbox("Auth Type", ["Digest", "Basic", "None"], index=1)
            
            # Capture button
            if st.button("📸 Capture Live Snapshot"):
                # Construct URL according to Axis VAPIX Video Streaming HTTP API
                snapshot_url = f"{cam_proto}://{cam_ip}:{cam_port}/axis-cgi/jpg/image.cgi?camera={cam_channel}&resolution={cam_res}"
                
                # Setup authentication handler
                auth_handler = None
                if cam_auth == "Digest" and cam_user and cam_pass:
                    auth_handler = HTTPDigestAuth(cam_user, cam_pass)
                elif cam_auth == "Basic" and cam_user and cam_pass:
                    auth_handler = (cam_user, cam_pass)
                
                status_box = st.info(f"Connecting to: `{snapshot_url}`...")
                
                try:
                    # Request snapshot with 6-second timeout
                    response = requests.get(snapshot_url, auth=auth_handler, timeout=6.0, verify=False)
                    
                    if response.status_code == 200:
                        try:
                            # Load response content bytes into PIL image
                            img_bytes = io.BytesIO(response.content)
                            img = Image.open(img_bytes).convert("RGB")
                            st.session_state.active_image = img
                            st.session_state.source_info = f"Axis Live Camera ({cam_ip}:{cam_port}, Ch:{cam_channel})"
                            status_box.empty()
                            st.toast("Live Axis snapshot captured successfully!", icon="📸")
                        except Exception as parse_err:
                            status_box.empty()
                            st.error(f"Axis API returned success, but response could not be parsed as an image: {parse_err}")
                            
                    elif response.status_code == 401:
                        status_box.empty()
                        st.error("❌ Authentication Failed: 401 Unauthorized.")
                        st.markdown("""
                        **Troubleshooting steps:**
                        1. Verify the configured **Username** and **Password**.
                        2. Verify if the camera requires **Digest** vs **Basic** authentication.
                        3. Check if the Axis user account has proper API/VAPIX Viewer privileges enabled.
                        """)
                    else:
                        status_box.empty()
                        st.error(f"❌ Camera returned HTTP Status Code: {response.status_code}")
                        st.markdown(f"**Response body snippet:** `{response.text[:200]}`")
                        
                except requests.exceptions.Timeout:
                    status_box.empty()
                    st.error("❌ Connection Timeout: The camera did not respond in time.")
                    st.markdown("""
                    **Troubleshooting steps:**
                    1. Check if the camera IP Address / Host (`{}`) is correct and reachable on your network.
                    2. Ping the camera host to test general connectivity.
                    3. Verify if port `{}` is correct and open.
                    """.format(cam_ip, cam_port))
                except requests.exceptions.ConnectionError as conn_err:
                    status_box.empty()
                    st.error(f"❌ Connection Error: {conn_err}")
                    st.markdown("""
                    **Troubleshooting steps:**
                    1. Double check cables and network settings.
                    2. Check if the VAPIX endpoint is enabled on this camera model.
                    """)
                except Exception as general_err:
                    status_box.empty()
                    st.error(f"❌ Unexpected Error: {general_err}")

# Results display column
with col_results:
    st.markdown("### 📊 Inspection Analysis")
    
    if st.session_state.active_image is None:
        st.info("Waiting for image input... Upload a produce file or capture a snapshot via the Axis camera integration tab.")
        
        # Display a placeholder container
        st.markdown("""
        <div style="border: 2px dashed #cbd5e1; border-radius: 12px; height: 300px; display: flex; align-items: center; justify-content: center; color: #64748b;">
            <div style="text-align: center;">
                <span style="font-size: 3rem;">🥦</span>
                <p style="margin-top: 10px; font-weight: 500;">No Active Inspection Image</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display image
        st.image(
            st.session_state.active_image, 
            caption=st.session_state.source_info, 
            use_container_width=True
        )
        
        # Process and Run Inference
        with st.spinner("Processing image & running neural networks..."):
            img = st.session_state.active_image
            # Preprocessing matching ONNX model requirements
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Run model
            try:
                input_name = session.get_inputs()[0].name
                pred = session.run(None, {input_name: img_array})[0][0][0]
                # Formula mapping logits/prob to 0-100 score
                freshness_score = (1.0 - pred) * 100.0
                
                # Keep score within reasonable bounds
                freshness_score = max(0.0, min(100.0, freshness_score))
            except Exception as inf_err:
                st.error(f"ONNX Model Inference failed: {inf_err}")
                st.stop()
                
        # Style details based on freshness score
        if freshness_score >= 60.0:
            status_text = "DISPATCH IMMEDIATELY"
            status_desc = "High freshness, prioritize for delivery"
            status_color = "#166534"
            status_bg = "#dcfce7"
            status_icon = "✅"
            action_text = "📦 Standard dispatch within 24 hours"
        elif freshness_score >= 25.0:
            status_text = "DISCOUNT & PRIORITIZE"
            status_desc = "Moderate freshness detected"
            status_color = "#854d0e"
            status_bg = "#fef9c3"
            status_icon = "⚠️"
            action_text = "💰 Apply 20–30% discount, dispatch today"
        else:
            status_text = "REMOVE FROM INVENTORY"
            status_desc = "Spoilage risk detected"
            status_color = "#991b1b"
            status_bg = "#fee2e2"
            status_icon = "❌"
            action_text = "🗑️ Flag for removal, do not dispatch"
            
        # Display metric card
        st.markdown(f"""
        <div style="background-color: {status_bg}; border-left: 6px solid {status_color}; padding: 1.2rem; border-radius: 8px; margin-bottom: 1.5rem;">
            <h4 style="color: {status_color}; margin: 0; font-weight: 600;">{status_icon} {status_text}</h4>
            <p style="color: {status_color}; opacity: 0.9; margin: 5px 0 0 0; font-size: 0.95rem;">{status_desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Metric
        col_score, col_action = st.columns([1, 1])
        with col_score:
            st.metric(
                label="Freshness Score", 
                value=f"{freshness_score:.1f} / 100",
                delta=f"{freshness_score - 50:.1f} vs Avg" if freshness_score > 50 else f"{freshness_score - 50:.1f} vs Avg",
                delta_color="normal"
            )
        with col_action:
            st.markdown("**Dispatch Directive:**")
            st.info(action_text)
            
        # Progress bar
        st.markdown("**Freshness Bar:**")
        st.progress(int(freshness_score))
        
        st.markdown("---")
        st.markdown("**Produce Supported:** Banana · Tomato · Potato")
        st.markdown("**Use Case:** Quality control intake check for quick commerce ops")
