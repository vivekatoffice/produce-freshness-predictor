import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os
import onnxruntime as ort

st.set_page_config(
    page_title="🍌 Produce Freshness Predictor",
    page_icon="🍌",
    layout="centered"
)

st.title("🍌 AI Freshness & Shelf-Life Predictor")
st.markdown("### Built for Quick-Commerce Platforms like Zepto & Blinkit")
st.write("Upload a banana image to get an instant freshness score and dispatch recommendation.")

@st.cache_resource
def load_model():
    if not os.path.exists('best_model.onnx'):
        gdown.download(
            'https://drive.google.com/uc?id=1i1rrXHPXoP-ckrwS0r4-PVpncmnaEB-P',
            'best_model.onnx',
            quiet=False,
            fuzzy=True
        )
    session = ort.InferenceSession('best_model.onnx')
    return session

session = load_model()

uploaded_file = st.file_uploader("📸 Upload produce image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    input_name = session.get_inputs()[0].name
    pred = session.run(None, {input_name: img_array})[0][0][0]
    freshness_score = (1 - pred) * 100

    st.markdown("---")
    st.metric(label="Freshness Score", value=f"{freshness_score:.1f} / 100")

    if freshness_score >= 75:
        st.success("✅ DISPATCH IMMEDIATELY — High freshness, prioritize for delivery")
        st.info("📦 Recommended Action: Standard dispatch within 24 hours")
    elif freshness_score >= 40:
        st.warning("⚠️ DISCOUNT & PRIORITIZE — Moderate freshness")
        st.info("💰 Recommended Action: Apply 20–30% discount, dispatch today")
    else:
        st.error("❌ REMOVE FROM INVENTORY — Spoilage risk detected")
        st.info("🗑️ Recommended Action: Flag for removal, do not dispatch")

    st.markdown("---")
    st.markdown("**Model:** MobileNetV2 Transfer Learning | **Accuracy:** 100% on 1,792 test images")
    st.markdown("**Use Case:** Automated quality control for fresh produce at warehouse intake")
