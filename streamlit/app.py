"""AI Waste Doctor browser frontend for Streamlit Cloud."""

import os
import sys
import time
from pathlib import Path

# Add repository root to Python import path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import requests
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from api_client import ClassificationError, classify_image
from live_scan import LiveScanState, LiveVideoProcessor

DEFAULT_API_URL = "https://ai-waste-doctor-api.onrender.com"
EXPECTED_API_BUILD = "2026-10-02-stable-live-lazy-semantic-v3.1"
CATEGORY_COLORS = {
    "Recyclable": "#FBBF24",
    "Dry Waste": "#60A5FA",
    "Wet Waste": "#22C55E",
}
CATEGORY_GUIDANCE = {
    "Recyclable": "BLUE RECYCLING BIN · Bottles, cans, glass, packaging, and electronics.",
    "Dry Waste": "YELLOW DRY-WASTE BIN · Wrappers, styrofoam, and non-recyclable packaging.",
    "Wet Waste": "GREEN COMPOST BIN · Food scraps and garden waste.",
}


def _get_secret(name: str):
    """Read a deployment secret from env or Streamlit Community Cloud secrets."""
    value = os.getenv(name)
    if value:
        return value.strip()
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value).strip() if value else None


def _tcp_turn_only(ice_servers):
    """Keep TURN/TCP/TLS URLs only and discard STUN/UDP candidates.

    Cloudflare returns STUN plus TURN/UDP/TCP/TLS endpoints. The Streamlit
    deployment has repeatedly shown aioice UDP datagram shutdown failures, so
    this app deliberately negotiates relay candidates over TCP only.
    """
    filtered = []
    preferred_order = (
        "turns:turn.cloudflare.com:443?transport=tcp",
        "turns:turn.cloudflare.com:5349?transport=tcp",
        "turn:turn.cloudflare.com:80?transport=tcp",
        "turn:turn.cloudflare.com:3478?transport=tcp",
    )

    for server in ice_servers or []:
        if not isinstance(server, dict):
            continue
        urls = server.get("urls", [])
        if isinstance(urls, str):
            urls = [urls]
        tcp_urls = [
            url
            for url in urls
            if isinstance(url, str)
            and url.startswith(("turn:", "turns:"))
            and "transport=tcp" in url
        ]
        if not tcp_urls:
            continue

        ordered = [url for wanted in preferred_order for url in tcp_urls if url == wanted]
        ordered.extend(url for url in tcp_urls if url not in ordered)
        item = {"urls": ordered}
        if server.get("username"):
            item["username"] = server["username"]
        if server.get("credential"):
            item["credential"] = server["credential"]
        filtered.append(item)

    return filtered


@st.cache_data(ttl=3300, show_spinner=False)
def get_rtc_configuration():
    """Build a relay-only WebRTC configuration that avoids UDP/STUN."""
    turn_key_id = _get_secret("CLOUDFLARE_TURN_KEY_ID")
    turn_api_token = _get_secret("CLOUDFLARE_TURN_KEY_API_TOKEN")

    if turn_key_id and turn_api_token:
        try:
            response = requests.post(
                f"https://rtc.live.cloudflare.com/v1/turn/keys/{turn_key_id}/credentials/generate-ice-servers",
                headers={
                    "Authorization": f"Bearer {turn_api_token}",
                    "User-Agent": "AI-Waste-Doctor/1.0 streamlit-webrtc",
                },
                json={"ttl": 7200},
                timeout=12,
            )
            response.raise_for_status()
            ice_servers = _tcp_turn_only(response.json().get("iceServers"))
            if ice_servers:
                return {
                    "iceServers": ice_servers,
                    "iceTransportPolicy": "relay",
                }, "Cloudflare TURN/TCP", None
            return None, "Cloudflare TURN/TCP", "Cloudflare returned no usable TURN/TCP servers."
        except (requests.RequestException, ValueError) as error:
            return None, "Cloudflare TURN/TCP", f"{type(error).__name__}: {error}"

    # Public fallback for testing. The science-fair deployment should normally
    # use Cloudflare TURN secrets, but this keeps the app usable without them.
    return (
        {
            "iceServers": [
                {
                    "urls": [
                        "turn:openrelay.metered.ca:443?transport=tcp",
                        "turn:openrelay.metered.ca:80?transport=tcp",
                    ],
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
                }
            ],
            "iceTransportPolicy": "relay",
        },
        "Open Relay TURN/TCP",
        None,
    )

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
    [data-testid="stSidebar"] { background: #2B2E38; border-right: 1px solid #454854; }
    [data-testid="stSidebarContent"] { padding-top: 2rem; }
    .app-header { display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: 1rem 1.25rem; border: 1px solid #454854; border-radius: 12px; background: #2B2E38; }
    .app-title { margin: 0; color: #B6FF2E; font-size: 1.65rem; line-height: 1.1; letter-spacing: .06em; font-weight: 900; }
    .app-subtitle { margin: .35rem 0 0; color: #94A3B8; font-size: .78rem; font-weight: 700; letter-spacing: .04em; }
    .mode-badge { padding: .55rem .8rem; border: 1px solid #B6FF2E; border-radius: 8px; color: #B6FF2E; font-size: .72rem; font-weight: 900; letter-spacing: .06em; white-space: nowrap; }
    .panel { padding: 1rem 1.1rem; border: 1px solid #454854; border-radius: 10px; background: #2B2E38; }
    .panel-label { margin: 0 0 .8rem; color: #38BDF8; font-size: .76rem; font-weight: 900; letter-spacing: .08em; }
    .decision { padding: 1.5rem 1rem; border: 2px solid #B6FF2E; border-radius: 10px; background: #343741; text-align: center; }
    .decision h2 { margin: 0; color: #B6FF2E; font-size: clamp(1.5rem, 3vw, 2.4rem); letter-spacing: .04em; }
    .decision p { margin: .55rem 0 0; color: #E2E8F0; }
    .decision.uncertain { border-color: #A78BFA; }
    .decision.uncertain h2 { color: #A78BFA; }
    .bar-label { display: flex; justify-content: space-between; margin-top: .75rem; color: #F1F5F9; font-size: .85rem; font-weight: 700; }
    .bar-track { height: 11px; margin-top: .28rem; border-radius: 6px; background: #252A35; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 6px; }
    .guidance { margin-top: 1rem; padding: .85rem 1rem; border-left: 4px solid #B6FF2E; background: #343741; color: #E2E8F0; font-size: .9rem; font-weight: 700; }
    .status-line { color: #94A3B8; font-size: .78rem; }
    .science-mode .decision { padding: 3rem 1rem; }
    .science-mode .decision h2 { font-size: clamp(2rem, 7vw, 4.8rem); }
    .science-header { display: flex; align-items: center; gap: .75rem; padding: .65rem .9rem; border: 2px solid #6B9E2A; border-radius: 12px; background: #2B2E38; color: #F1F5F9; }
    .science-title { flex: 1; margin: 0; color: #B6FF2E; font-size: 1.35rem; font-weight: 900; letter-spacing: .08em; }
    .science-chip { padding: .55rem .7rem; border: 1px solid #6B9E2A; border-radius: 8px; color: #38BDF8; background: #343741; font-size: .7rem; font-weight: 800; white-space: nowrap; }
    .science-panel-title { margin: .9rem 0 .5rem; color: #38BDF8; font-size: .75rem; font-weight: 900; letter-spacing: .08em; }
    .science-video-panel { min-height: 510px; padding: .65rem; border: 2px solid #D6E3DC; border-radius: 10px; background: #F7F8F5; }
    .science-result-panel { min-height: 510px; padding: .8rem; border: 2px solid #D6E3DC; border-radius: 10px; background: #F7F8F5; color: #1F2937; }
    .science-decision { padding: 1.4rem .9rem; border: 3px solid #B6FF2E; border-radius: 12px; background: #343741; text-align: center; }
    .science-decision h2 { margin: 0; color: #B6FF2E; font-size: clamp(1.7rem, 3vw, 2.8rem); letter-spacing: .05em; }
    .science-decision p { margin: .7rem 0 0; color: #38BDF8; font-size: 1rem; font-weight: 800; }
    .science-decision.uncertain { border-color: #A78BFA; }
    .science-decision.uncertain h2 { color: #A78BFA; }
    .science-bar-label { display: flex; justify-content: space-between; margin-top: .7rem; color: #F1F5F9; font-size: .8rem; font-weight: 800; }
    .science-bar-track { height: 12px; margin-top: .25rem; border-radius: 7px; background: #252A35; overflow: hidden; }
    .science-bar-fill { height: 100%; border-radius: 7px; }
    .science-status { margin: .8rem 0; padding: .65rem .8rem; border-left: 4px solid #B6FF2E; background: #343741; color: #E2E8F0; font-weight: 800; }
    .science-footer { color: #94A3B8; font-size: .75rem; text-align: center; }
    [data-testid="stAppViewContainer"]:fullscreen,
    [data-testid="stAppViewContainer"]:fullscreen::backdrop,
    body:has([data-testid="stAppViewContainer"]:fullscreen) {
        background: #23262F !important;
    }
    /* In SENDONLY mode the component's video area is intentionally not used.
       Keep only a compact strip for START/STOP and device controls; the actual
       preview is rendered below from frames received by the Python processor. */
    iframe[src*="streamlit_webrtc"] { width: 100% !important; min-width: 100% !important; height: 145px !important; min-height: 145px !important; aspect-ratio: auto !important; display: block; border: 0; background: #0F131D; }
    [data-testid="column"]:has(iframe[src*="streamlit_webrtc"]) { min-width: 0 !important; width: 100% !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "api_url" not in st.session_state:
    st.session_state["api_url"] = os.getenv("AI_WASTE_API_URL", DEFAULT_API_URL).rstrip("/")
if "result" not in st.session_state:
    st.session_state["result"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []
if "live_scan_state" not in st.session_state:
    st.session_state["live_scan_state"] = LiveScanState()
if "live_last_inference" not in st.session_state:
    st.session_state["live_last_inference"] = 0.0
if "live_last_frame_at" not in st.session_state:
    st.session_state["live_last_frame_at"] = 0.0
if "live_inference_in_flight" not in st.session_state:
    st.session_state["live_inference_in_flight"] = False

api_url = st.session_state["api_url"]
science_mode = st.sidebar.toggle("SCIENCE FAIR MODE", value=False)
st.sidebar.markdown("### SYSTEM STATUS")
st.sidebar.caption(f"API: {api_url}")

st.markdown(
    f"""
    <div class="app-header">
      <div>
        <p class="app-title">AI WASTE DOCTOR</p>
        <p class="app-subtitle">INTELLIGENT WASTE CLASSIFICATION SYSTEM · RECYCLE · SORT · CLEANER PLANET</p>
      </div>
      <div class="mode-badge">{'SCIENCE FAIR MODE' if science_mode else 'ONLINE MODE'}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


def classify_uploaded_file(uploaded_file, status_box=None):
    """Classify a Streamlit upload or browser camera capture through the API."""
    return classify_image(
        api_url,
        uploaded_file.name or "camera-capture.jpg",
        uploaded_file.getvalue(),
        uploaded_file.type or "image/jpeg",
        on_status=status_box.caption if status_box else None,
    )


def show_classification_error(error):
    """Show a concise user message while keeping diagnostics available."""
    st.error(error.user_message)
    with st.expander("Technical details"):
        st.code(error.technical)




def render_live_inference(ctx):
    """Render a stable manual camera scanner without Streamlit fragment reruns.

    WebRTC callbacks run independently from Streamlit's script lifecycle.
    Keeping the scan action in the normal Streamlit script prevents stale
    fragment callbacks from accessing removed session-state/component keys.
    """
    processor = ctx.video_processor if ctx is not None else None
    if processor is None:
        st.info("Start the camera to begin live scanning.")
        return

    frame, captured_at = processor.latest_frame()
    width, height, camera_fps = processor.frame_info()

    if width and height:
        st.caption(
            f"CAMERA: {width}×{height} · {camera_fps:.1f} FPS · "
            "MODE: MANUAL STABLE SCAN"
        )

    if frame is not None:
        preview = frame.copy()
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        st.image(preview, channels="RGB", width="stretch")
    else:
        st.info("Waiting for the first camera frame...")

    col_scan, col_reset = st.columns(2)
    with col_scan:
        scan_clicked = st.button(
            "SCAN CURRENT OBJECT",
            key="science_scan_current_v6",
            type="primary",
            width="stretch",
        )
    with col_reset:
        reset_clicked = st.button(
            "RESET DECISION",
            key="science_scan_reset_v6",
            width="stretch",
        )

    live_state = st.session_state["live_scan_state"]

    if reset_clicked:
        live_state.reset()
        st.session_state["live_last_inference"] = 0.0
        st.rerun()

    if scan_clicked:
        if frame is None:
            st.warning("Camera frame is not ready yet. Wait a moment and scan again.")
        else:
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 88],
            )
            if not ok:
                st.error("Could not encode the current camera frame.")
            else:
                try:
                    result = classify_image(
                        api_url,
                        "live-frame.jpg",
                        encoded.tobytes(),
                        "image/jpeg",
                        max_attempts=2,
                    )
                    snapshot = live_state.update(result)
                    st.session_state["live_last_inference"] = time.monotonic()
                    if snapshot.is_final:
                        st.toast(f"DECISION COMMITTED · {snapshot.top_class}")
                    else:
                        st.info(
                            f"Stability check: {snapshot.consecutive_frames}/3 frames · "
                            f"{snapshot.top_class} {snapshot.confidence:.1f}%"
                        )
                except ClassificationError as error:
                    show_classification_error(error)

    result = live_state.display_result()
    final_class = live_state.final_class

    if result:
        render_science_prediction(result, final_class)
    else:
        st.markdown(
            '<div class="science-decision uncertain">'
            '<h2>READY TO SCAN</h2>'
            '<p>Place ONE object inside the green square, then press SCAN CURRENT OBJECT.</p>'
            '</div>',
            unsafe_allow_html=True,
        )


def render_science_prediction(result, final_class):
    """Render the desktop-style science-fair prediction panel."""
    predictions = result.get("predictions", {})
    confidence = float(result.get("top_confidence", 0.0))
    predicted = final_class or result.get("top_class", "Scanning")
    is_final = bool(final_class)
    css_class = "" if is_final else "uncertain"
    headline = f"FINAL: {predicted.upper()}" if is_final else "SCANNING OBJECT"
    message = f"{confidence:.1f}% CONFIDENCE" if is_final else f"LIVE PROBABILITY · {confidence:.1f}% TOP SIGNAL"
    st.markdown(
        f'<div class="science-decision {css_class}"><h2>{headline}</h2><p>{message}</p></div>',
        unsafe_allow_html=True,
    )
    if is_final:
        st.markdown(f'<div class="science-status">{CATEGORY_GUIDANCE.get(predicted, "Place this item in the designated sorting bin.")}</div>', unsafe_allow_html=True)

    semantic_hint = result.get("semantic_hint") or {}
    if result.get("semantic_guard_applied"):
        semantic_label = semantic_hint.get("label") or "food / produce"
        semantic_confidence = float(semantic_hint.get("confidence", 0.0))
        produce_mass = float(semantic_hint.get("produce_mass", 0.0))
        st.success(
            f"FOOD CROSS-CHECK: {semantic_label} detected "
            f"({semantic_confidence:.1f}% strongest cue · {produce_mass:.1f}% produce evidence)"
        )

    st.markdown("<h4 style='color:#334155;margin:.8rem 0 .2rem'>CLASS CONFIDENCE PROBABILITIES</h4>", unsafe_allow_html=True)
    for category, color in CATEGORY_COLORS.items():
        value = max(0.0, min(100.0, float(predictions.get(category, 0.0))))
        st.markdown(
            f'<div class="science-bar-label"><span>{category}</span><span>{value:.1f}%</span></div>'
            f'<div class="science-bar-track"><div class="science-bar-fill" style="width:{value}%;background:{color};"></div></div>',
            unsafe_allow_html=True,
        )
    st.caption(f"Model build: {result.get('build', 'not reported')} · Decision threshold: 70% · Stable frames: 3")


def render_prediction(result):
    """Render the shared prediction panel used by upload and camera tabs."""
    if not result:
        st.markdown('<div class="decision"><h2>READY FOR SCAN</h2><p>Choose an image or use your camera.</p></div>', unsafe_allow_html=True)
        predictions = {name: 0.0 for name in CATEGORY_COLORS}
        return

    predicted = result.get("top_class", "Uncertain")
    confidence = float(result.get("top_confidence", 0.0))
    confident = bool(result.get("is_confident", False))
    headline = predicted.upper() if confident else "UNCERTAIN"
    message = f"FINAL DECISION · {confidence:.1f}% CONFIDENCE" if confident else f"NO BIN DECISION · TOP SIGNAL {predicted.upper()} ({confidence:.1f}%)"
    css_class = "" if confident else "uncertain"
    st.markdown(f'<div class="decision {css_class}"><h2>{headline}</h2><p>{message}</p></div>', unsafe_allow_html=True)
    predictions = result.get("predictions", {})
    if confident:
        st.markdown(f'<div class="guidance">{CATEGORY_GUIDANCE.get(predicted, "Place this item in the designated sorting bin.")}</div>', unsafe_allow_html=True)
    result_build = result.get("build")
    if result_build != EXPECTED_API_BUILD:
        st.warning("The classification API is running an older model build. Redeploy the API before trusting this result.")
    st.caption(f"Model build: {result_build or 'not reported'}")

    semantic_hint = result.get("semantic_hint") or {}
    if result.get("semantic_guard_applied"):
        semantic_label = semantic_hint.get("label") or "food / produce"
        st.info(
            f"Food cross-check corrected this result to Wet Waste · "
            f"cue: {semantic_label} · "
            f"produce evidence: {float(semantic_hint.get('produce_mass', 0.0)):.1f}%"
        )

    st.markdown("#### CONFIDENCE BREAKDOWN")
    for category, color in CATEGORY_COLORS.items():
        value = max(0.0, min(100.0, float(predictions.get(category, 0.0))))
        st.markdown(
            f'<div class="bar-label"><span>{category}</span><span>{value:.1f}%</span></div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{value}%;background:{color};"></div></div>',
            unsafe_allow_html=True,
        )


navigation_options = ["📁  UPLOAD IMAGE", "📹  LIVE CAMERA", "🎪  SCIENCE FAIR", "💡  HOW IT WORKS", "⚙️  SETTINGS"]
active_view = st.radio(
    "Application view",
    navigation_options,
    horizontal=True,
    label_visibility="collapsed",
    key="active_view",
)

if active_view == navigation_options[0]:
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown('<p class="panel-label">IMAGE SCAN</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a waste image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        if uploaded_file:
            scan_clicked = st.button("CLASSIFY UPLOADED IMAGE", type="primary", width="stretch")
            st.image(uploaded_file, caption="OBJECT READY FOR CLASSIFICATION", width="stretch")
            if scan_clicked:
                status_box = st.empty()
                with st.spinner("Analyzing object..."):
                    try:
                        st.session_state["result"] = classify_uploaded_file(uploaded_file, status_box)
                        st.session_state["history"].append(st.session_state["result"])
                    except ClassificationError as error:
                        show_classification_error(error)
                status_box.empty()
        else:
            st.info("Place one object clearly in the image for the most reliable result.")
    with right:
        st.markdown('<p class="panel-label">AI PREDICTION</p>', unsafe_allow_html=True)
        render_prediction(st.session_state["result"])

if active_view == navigation_options[1]:
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown('<p class="panel-label">CAMERA SCAN</p>', unsafe_allow_html=True)
        camera_file = st.camera_input("Capture a waste object", label_visibility="collapsed")
        if camera_file and st.button("CLASSIFY CAPTURE", type="primary", width="stretch"):
            status_box = st.empty()
            with st.spinner("Analyzing captured object..."):
                try:
                    st.session_state["result"] = classify_uploaded_file(camera_file, status_box)
                    st.session_state["history"].append(st.session_state["result"])
                except ClassificationError as error:
                    show_classification_error(error)
            status_box.empty()
    with right:
        st.markdown('<p class="panel-label">LIVE RESULT</p>', unsafe_allow_html=True)
        render_prediction(st.session_state["result"])

if active_view == navigation_options[2]:
    st.markdown(
        '<div class="science-header"><p class="science-title">AI WASTE DOCTOR</p>'
        '<span class="science-chip">SCAN → CLASSIFY → SORT</span>'
        '<span class="science-chip">LIGHT: SIMULATION</span>'
        '<span class="science-chip">METAL: NOT CONNECTED</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="science-footer">Science Fair Live Mode · camera video is continuous · inference samples one frame per second</p>', unsafe_allow_html=True)
    st.iframe(
        """
        <button
          onclick="(() => {
            try {
              const root = window.parent.document.querySelector('[data-testid=stAppViewContainer]') || window.parent.document.documentElement;
              if (root.requestFullscreen) {
                root.requestFullscreen().catch(() =>
                  window.parent.alert('Fullscreen was blocked. Use browser fullscreen or allow fullscreen for this page.')
                );
              }
            } catch (error) {
              window.parent.alert('Fullscreen requires HTTPS or localhost and browser permission.');
            }
          })()"
          style="padding:8px 14px;font-weight:700;cursor:pointer;border-radius:8px;border:1px solid #6B9E2A;background:#343741;color:#B6FF2E"
        >ENTER FULL SCREEN</button>
        """,
        height=55,
        width=240,
    )

    left, right = st.columns([1.08, .92], gap="medium")
    with left:
        st.markdown('<p class="science-panel-title">LIVE CAMERA FEED</p>', unsafe_allow_html=True)
        try:
            rtc_configuration, rtc_mode, rtc_error = get_rtc_configuration()
            if rtc_error:
                st.error(
                    "TURN configuration failed. "
                    f"{rtc_error}"
                )
            elif rtc_mode == "Open Relay TURN/TCP":
                st.caption(
                    "NETWORK: TURN relay over TCP · one-way camera transport"
                )
            else:
                st.caption("NETWORK: Cloudflare TURN/TCP relay · one-way camera transport")

            ctx = webrtc_streamer(
                key="science-fair-camera-v5-stable",
                mode=WebRtcMode.SENDONLY,
                rtc_configuration=rtc_configuration,
                video_processor_factory=LiveVideoProcessor,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640, "min": 320},
                        "height": {"ideal": 480, "min": 240},
                        "frameRate": {"ideal": 15, "max": 20},
                    },
                    "audio": False,
                },
                sendback_audio=False,
                media_toggle_controls=False,
                async_processing=True,
            )

            if ctx is not None and ctx.state.playing:
                pass
            else:
                st.caption("Press START, choose a camera if needed, and allow browser camera permission.")
        except Exception as error:
            ctx = None
            st.warning("Live browser video is unavailable in this session. Use the LIVE CAMERA capture tab instead.")
            st.caption(f"WebRTC status: {type(error).__name__}")
    with right:
        st.markdown('<p class="science-panel-title">REAL-TIME AI CLASSIFICATION</p>', unsafe_allow_html=True)
        if ctx is None or not ctx.state.playing:
            st.markdown('<div class="science-decision uncertain"><h2>SCANNING OBJECT</h2><p>START CAMERA TO BEGIN</p></div>', unsafe_allow_html=True)
            st.caption("Press START and allow camera access. The live stream uses TURN over TCP/443 on hosted deployments.")
        else:
            render_live_inference(ctx)

if active_view == navigation_options[3]:
    st.markdown('<p class="panel-label">HOW IT WORKS</p>', unsafe_allow_html=True)
    steps = [
        ("01", "CAPTURE", "Upload an image or take a photo of one waste object."),
        ("02", "PREPROCESS", "The service resizes and normalizes the image for the trained model."),
        ("03", "CLASSIFY", "The AI compares the object with Recyclable, Dry Waste, and Wet Waste classes."),
        ("04", "SORT", "Use the confidence result and disposal guidance to choose the right bin."),
    ]
    for number, title, text in steps:
        col_number, col_copy = st.columns([.12, .88])
        with col_number:
            st.markdown(f"### {number}")
        with col_copy:
            st.markdown(f"**{title}**  \n{text}")
        st.divider()

if active_view == navigation_options[4]:
    st.markdown('<p class="panel-label">SYSTEM SETTINGS</p>', unsafe_allow_html=True)
    configured_url = st.text_input("Classification API URL", value=api_url)
    if st.button("SAVE API SETTINGS", type="primary"):
        st.session_state["api_url"] = configured_url.rstrip("/")
        st.success("API endpoint saved for this browser session.")
        st.rerun()
    if st.button("CHECK API STATUS"):
        try:
            response = requests.get(f"{api_url}/health", timeout=30)
            content_type = response.headers.get("Content-Type", "").lower()
            health_data = response.json() if "json" in content_type and response.content else {}
            if response.ok and health_data.get("model_loaded"):
                st.success("ONLINE · Model loaded and ready for classification.")
            else:
                st.warning(f"SERVICE NOT READY · HTTP {response.status_code}")
                if health_data.get("model_error") or health_data.get("model_status"):
                    st.caption(health_data.get("model_error") or health_data.get("model_status"))
            if health_data.get("build"):
                st.caption(f"Model build: {health_data['build']}")
            semantic_status = health_data.get("semantic_guard_status")
            if semantic_status:
                st.caption(f"Food cross-check: {semantic_status}")
        except ValueError:
            st.error(f"API returned a non-JSON response · HTTP {response.status_code}")
        except requests.RequestException as error:
            st.error(f"API unavailable: {error}")
    st.markdown(f'<p class="status-line">Scans this session: {len(st.session_state["history"])} · API: {api_url}</p>', unsafe_allow_html=True)

st.caption(f"Status: {'SCIENCE FAIR DISPLAY' if science_mode else 'ONLINE MODE'}  |  Categories: {' | '.join(CATEGORY_COLORS)}  |  API: {api_url}")