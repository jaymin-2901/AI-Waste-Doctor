"""AI Waste Doctor browser frontend for Streamlit Cloud."""

import os

import requests
import streamlit as st

API_URL = os.getenv("AI_WASTE_API_URL", "https://ai-waste-doctor-api.onrender.com").rstrip("/")
CATEGORY_COLORS = {
    "Recyclable": "#D97706",
    "Dry Waste": "#2563EB",
    "Wet Waste": "#16A34A",
}

st.set_page_config(
    page_title="AI Waste Doctor",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #23262F; color: #F1F5F9; }
    [data-testid="stHeader"] { background: transparent; }
    .brand { padding: 1.2rem 1.4rem; border: 1px solid #3B404D; border-radius: 16px; background: #2B2E38; }
    .brand h1 { margin: 0; color: #B6FF2E; font-size: 2rem; letter-spacing: .04em; }
    .brand p { margin: .35rem 0 0; color: #B5BFCD; }
    .panel { padding: 1.2rem; border: 1px solid #3B404D; border-radius: 16px; background: #2B2E38; min-height: 220px; }
    .decision { padding: 1.4rem; border: 2px solid #B6FF2E; border-radius: 16px; background: #343741; text-align: center; }
    .decision h2 { margin: 0; color: #B6FF2E; }
    .decision p { color: #E2E8F0; margin-bottom: 0; }
    .uncertain { border-color: #A78BFA; }
    .uncertain h2 { color: #A78BFA; }
    .bar-label { display: flex; justify-content: space-between; margin-top: .7rem; color: #F1F5F9; font-weight: 700; }
    .bar-track { height: 13px; margin-top: .25rem; border-radius: 8px; background: #252A35; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 8px; }
    .caption { color: #AAB4C2; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="brand">
      <h1>AI WASTE DOCTOR</h1>
      <p>Intelligent waste classification · Recyclable · Dry Waste · Wet Waste</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

upload_col, result_col = st.columns([1.05, 0.95], gap="large")

with upload_col:
    st.subheader("Scan an object")
    source = st.radio("Input source", ["Upload image", "Use camera"], horizontal=True, label_visibility="collapsed")
    image_file = st.file_uploader(
        "Choose a waste image",
        type=["jpg", "jpeg", "png", "webp"],
        disabled=source != "Upload image",
    )
    camera_file = st.camera_input("Take a photo", disabled=source != "Use camera")
    selected_file = image_file if source == "Upload image" else camera_file

    if selected_file:
        st.image(selected_file, caption="Object ready for classification", use_container_width=True)
        if st.button("CLASSIFY OBJECT", type="primary", use_container_width=True):
            with st.spinner("Analyzing object..."):
                try:
                    health = requests.get(f"{API_URL}/health", timeout=30)
                    health.raise_for_status()
                    health_data = health.json()
                    if not health_data.get("model_loaded", False):
                        detail = health_data.get("model_error") or health_data.get("model_status") or "Model is not loaded"
                        st.error(f"The Render service is online but its model is unavailable: {detail}")
                        st.stop()
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (selected_file.name, selected_file.getvalue(), selected_file.type)},
                        timeout=120,
                    )
                    response.raise_for_status()
                    st.session_state["result"] = response.json()
                except requests.RequestException as error:
                    st.error(f"Classification service unavailable: {error}")
    else:
        st.markdown('<p class="caption">Place one object clearly in the image for the most reliable result.</p>', unsafe_allow_html=True)

with result_col:
    st.subheader("AI prediction")
    result = st.session_state.get("result")
    if not result:
        st.markdown('<div class="decision"><h2>READY FOR SCAN</h2><p>Upload an image or use your camera.</p></div>', unsafe_allow_html=True)
        predictions = {name: 0.0 for name in CATEGORY_COLORS}
    else:
        predicted = result.get("top_class", "Uncertain")
        confidence = float(result.get("top_confidence", 0.0))
        confident = bool(result.get("is_confident", False))
        css_class = "" if confident else "uncertain"
        headline = predicted.upper() if confident else "UNCERTAIN"
        message = f"FINAL DECISION · {confidence:.1f}% confidence" if confident else "Move the object closer or improve lighting"
        st.markdown(f'<div class="decision {css_class}"><h2>{headline}</h2><p>{message}</p></div>', unsafe_allow_html=True)
        predictions = result.get("predictions", {})

    st.markdown("### Class confidence probabilities")
    for category, color in CATEGORY_COLORS.items():
        value = max(0.0, min(100.0, float(predictions.get(category, 0.0))))
        st.markdown(
            f'<div class="bar-label"><span>{category}</span><span>{value:.1f}%</span></div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{value}%;background:{color};"></div></div>',
            unsafe_allow_html=True,
        )

st.caption(f"Classification API: {API_URL}")
