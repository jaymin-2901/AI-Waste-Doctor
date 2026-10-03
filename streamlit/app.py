"""AI Waste Doctor browser frontend for Streamlit Cloud."""

import io
import math
import os
import sys
import time
import wave
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
EXPECTED_API_BUILD = "2026-10-03-science-fair-material-guard-v5"
LIVE_INFERENCE_INTERVAL = 0.90
LIVE_INITIAL_SETTLE_SECONDS = 0.75
LIVE_DECISION_HOLD_SECONDS = 1.0
LIVE_SCENE_CHANGE_THRESHOLD = 18.0
LIVE_SCENE_CHANGE_HITS = 3
LIVE_ERROR_COOLDOWN = 6.0
LIVE_PREDICT_TIMEOUT = 30
LIVE_INFERENCE_INTERVAL = 0.90
LIVE_INITIAL_SETTLE_SECONDS = 0.75
LIVE_DECISION_HOLD_SECONDS = 1.0
LIVE_SCENE_CHANGE_THRESHOLD = 18.0
LIVE_SCENE_CHANGE_HITS = 3
LIVE_ERROR_COOLDOWN = 6.0
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


def _is_mobile_browser():
    """Return True for common phone/tablet user agents."""
    try:
        user_agent = str(st.context.headers.get("User-Agent", "")).lower()
    except Exception:
        user_agent = ""
    mobile_tokens = (
        "android",
        "iphone",
        "ipad",
        "ipod",
        "mobile",
        "windows phone",
        "opera mini",
        "iemobile",
    )
    return any(token in user_agent for token in mobile_tokens)


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
    :root {
        color-scheme: dark;
        --bg: #0b1118;
        --surface: #111923;
        --surface-2: #17212d;
        --surface-3: #1d2a37;
        --border: rgba(148,163,184,.16);
        --border-strong: rgba(182,255,46,.30);
        --text: #f8fafc;
        --muted: #94a3b8;
        --accent: #b6ff2e;
        --cyan: #38bdf8;
        --violet: #a78bfa;
        --shadow: 0 18px 50px rgba(0,0,0,.22);
        --radius: 18px;
    }

    * { box-sizing: border-box; }
    html, body { background: var(--bg) !important; }
    .stApp {
        background:
            radial-gradient(circle at 8% -10%, rgba(182,255,46,.08), transparent 28%),
            radial-gradient(circle at 92% 0%, rgba(56,189,248,.07), transparent 25%),
            var(--bg);
        color: var(--text);
    }
    [data-testid="stAppViewContainer"] > .main {
        padding: 1.25rem clamp(.75rem, 3vw, 2.5rem) 2rem;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: .5rem; }
    [data-testid="stSidebar"] {
        background: rgba(13,20,29,.96);
        border-right: 1px solid var(--border);
        backdrop-filter: blur(18px);
    }
    [data-testid="stSidebarContent"] { padding: 1.25rem; }

    .app-header {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: clamp(1rem, 2vw, 1.5rem) clamp(1rem, 3vw, 1.75rem);
        border: 1px solid var(--border);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(23,33,45,.96), rgba(17,25,35,.96));
        box-shadow: var(--shadow);
        overflow: hidden;
    }
    .app-header::after {
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        right: -70px;
        top: -90px;
        border-radius: 50%;
        background: rgba(182,255,46,.08);
        pointer-events: none;
    }
    .app-title {
        margin: 0;
        color: var(--accent);
        font-size: clamp(1.35rem, 3vw, 2rem);
        line-height: 1;
        letter-spacing: .08em;
        font-weight: 950;
    }
    .app-subtitle {
        margin: .5rem 0 0;
        max-width: 850px;
        color: var(--muted);
        font-size: clamp(.66rem, 1.1vw, .78rem);
        font-weight: 700;
        letter-spacing: .045em;
        line-height: 1.55;
    }
    .mode-badge {
        z-index: 1;
        flex: 0 0 auto;
        padding: .65rem .9rem;
        border: 1px solid rgba(182,255,46,.55);
        border-radius: 999px;
        color: var(--accent);
        background: rgba(182,255,46,.06);
        font-size: .68rem;
        font-weight: 900;
        letter-spacing: .08em;
        white-space: nowrap;
    }

    /* Main navigation: desktop pills, horizontal scroll on small screens. */
    [data-testid="stRadio"] {
        margin: 1.05rem 0 .8rem;
    }
    [data-testid="stRadio"] > label {
        display: none;
    }
    [data-testid="stRadio"] [role="radiogroup"] {
        display: flex;
        flex-wrap: nowrap;
        gap: .5rem;
        width: 100%;
        padding: .35rem;
        overflow-x: auto;
        scrollbar-width: none;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: rgba(17,25,35,.82);
        box-shadow: 0 8px 28px rgba(0,0,0,.12);
    }
    [data-testid="stRadio"] [role="radiogroup"]::-webkit-scrollbar { display: none; }
    [data-testid="stRadio"] [role="radiogroup"] label {
        position: relative;
        flex: 1 1 0;
        min-height: 46px;
        min-width: 0;
        margin: 0 !important;
        padding: .68rem .7rem !important;
        border: 1px solid transparent;
        border-radius: 11px;
        color: #f8fafc !important;
        background: #18222d !important;
        transition: .18s ease;
        overflow: hidden;
    }
    [data-testid="stRadio"] [role="radiogroup"] label:hover {
        color: #ffffff !important;
        background: #22303d !important;
        border-color: rgba(255,255,255,.10);
    }
    [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        color: #071008 !important;
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        box-shadow: 0 7px 20px rgba(182,255,46,.18);
        font-weight: 950;
    }
    [data-testid="stRadio"] [role="radiogroup"] label p,
    [data-testid="stRadio"] [role="radiogroup"] label span {
        color: inherit !important;
        font-size: .76rem !important;
        font-weight: 900 !important;
        white-space: nowrap;
    }
    [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p,
    [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) span {
        color: #071008 !important;
    }
    [data-testid="stRadio"] [role="radiogroup"] input {
        position: absolute !important;
        opacity: 0 !important;
        width: 1px !important;
        height: 1px !important;
        pointer-events: none !important;
    }

    .panel {
        padding: clamp(1rem, 2vw, 1.35rem);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        background: linear-gradient(145deg, rgba(23,33,45,.94), rgba(15,23,32,.94));
        box-shadow: var(--shadow);
    }
    .panel-label, .science-panel-title {
        margin: .15rem 0 .8rem;
        color: var(--cyan);
        font-size: .72rem;
        font-weight: 950;
        letter-spacing: .12em;
        text-transform: uppercase;
    }

    /* Streamlit column blocks become true single-column cards on phones. */
    [data-testid="stHorizontalBlock"] {
        gap: clamp(.75rem, 2vw, 1.5rem);
        align-items: stretch;
    }
    [data-testid="column"] {
        min-width: 0 !important;
    }

    /* Inputs and action buttons. */
    [data-testid="stFileUploaderDropzone"] {
        min-height: 150px;
        border: 1px dashed rgba(148,163,184,.30) !important;
        border-radius: 16px !important;
        background: rgba(255,255,255,.025) !important;
    }
    [data-testid="stCameraInput"] {
        overflow: hidden;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: rgba(255,255,255,.025);
    }
    .stButton > button,
    [data-testid="stDownloadButton"] > button {
        min-height: 44px;
        border-radius: 12px !important;
        font-weight: 850 !important;
        letter-spacing: .025em;
        transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
    }
    .stButton > button:hover,
    [data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] {
        color: #0b1118 !important;
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        box-shadow: 0 9px 24px rgba(182,255,46,.16);
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 12px 30px rgba(182,255,46,.24);
    }
    [data-testid="stAlert"] {
        border-radius: 13px !important;
    }

    .decision {
        min-height: 190px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 1.5rem 1rem;
        border: 1px solid rgba(182,255,46,.45);
        border-radius: 18px;
        background:
            radial-gradient(circle at 50% 0%, rgba(182,255,46,.09), transparent 50%),
            var(--surface-2);
        text-align: center;
        box-shadow: var(--shadow);
    }
    .decision h2 {
        margin: 0;
        color: var(--accent);
        font-size: clamp(1.5rem, 4vw, 2.65rem);
        letter-spacing: .035em;
        line-height: 1.05;
    }
    .decision p { margin: .65rem 0 0; color: #dbe4ee; }
    .decision.uncertain {
        border-color: rgba(167,139,250,.55);
        background: radial-gradient(circle at 50% 0%, rgba(167,139,250,.09), transparent 52%), var(--surface-2);
    }
    .decision.uncertain h2 { color: var(--violet); }

    .bar-label {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-top: .8rem;
        color: #e8eef5;
        font-size: .82rem;
        font-weight: 750;
    }
    .bar-track {
        height: 9px;
        margin-top: .3rem;
        border-radius: 999px;
        background: #26313e;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: inherit;
        transition: width .35s ease;
    }
    .guidance {
        margin-top: 1rem;
        padding: .9rem 1rem;
        border: 1px solid rgba(182,255,46,.14);
        border-left: 4px solid var(--accent);
        border-radius: 12px;
        background: rgba(182,255,46,.055);
        color: #e2e8f0;
        font-size: .88rem;
        font-weight: 750;
        line-height: 1.55;
    }
    .status-line, .science-footer {
        color: var(--muted);
        font-size: .75rem;
        line-height: 1.5;
    }

    /* Science-fair presentation mode. */
    .science-header {
        display: flex;
        align-items: center;
        gap: .55rem;
        padding: .8rem;
        border: 1px solid var(--border-strong);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(23,33,45,.98), rgba(17,25,35,.98));
        box-shadow: var(--shadow);
        flex-wrap: wrap;
    }
    .science-title {
        flex: 1 1 220px;
        margin: 0;
        color: var(--accent);
        font-size: clamp(1.1rem, 2.5vw, 1.45rem);
        font-weight: 950;
        letter-spacing: .08em;
    }
    .science-chip {
        padding: .5rem .65rem;
        border: 1px solid rgba(107,158,42,.5);
        border-radius: 999px;
        color: var(--cyan);
        background: rgba(52,55,65,.65);
        font-size: .65rem;
        font-weight: 850;
        white-space: nowrap;
    }
    .science-video-panel, .science-result-panel {
        min-height: 510px;
        padding: .7rem;
        border: 1px solid rgba(214,227,220,.18);
        border-radius: 18px;
        background: #f7f8f5;
        box-shadow: var(--shadow);
    }
    .science-decision {
        min-height: 185px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 1.4rem .9rem;
        border: 2px solid var(--accent);
        border-radius: 18px;
        background: #17212d;
        text-align: center;
    }
    .science-decision h2 {
        margin: 0;
        color: var(--accent);
        font-size: clamp(1.55rem, 4vw, 2.9rem);
        letter-spacing: .045em;
        line-height: 1.05;
    }
    .science-decision p {
        margin: .7rem 0 0;
        color: var(--cyan);
        font-size: .95rem;
        font-weight: 850;
    }
    .science-decision.uncertain {
        border-color: rgba(167,139,250,.7);
    }
    .science-decision.uncertain h2 { color: var(--violet); }
    .science-bar-label {
        display: flex;
        justify-content: space-between;
        margin-top: .7rem;
        color: #f1f5f9;
        font-size: .78rem;
        font-weight: 800;
    }
    .science-bar-track {
        height: 10px;
        margin-top: .25rem;
        border-radius: 999px;
        background: #252a35;
        overflow: hidden;
    }
    .science-bar-fill { height: 100%; border-radius: inherit; }
    .science-status {
        margin: .8rem 0;
        padding: .75rem .85rem;
        border: 1px solid rgba(182,255,46,.13);
        border-left: 4px solid var(--accent);
        border-radius: 12px;
        background: #343741;
        color: #e2e8f0;
        font-weight: 800;
        line-height: 1.5;
    }

    .science-camera-toolbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: .65rem;
        margin: 0 0 .8rem;
        padding: .65rem .75rem;
        border: 1px solid rgba(182,255,46,.22);
        border-radius: 14px;
        background: #111923;
    }
    .science-camera-status {
        color: #f8fafc;
        font-size: .72rem;
        font-weight: 850;
        letter-spacing: .035em;
    }
    .science-camera-status strong {
        color: var(--accent);
    }
    .science-header .science-chip {
        color: #f8fafc;
        background: #18222d;
        border-color: rgba(255,255,255,.18);
    }
    .science-header .science-chip:first-of-type {
        color: #071008;
        background: var(--accent);
        border-color: var(--accent);
    }

    .science-legend {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: .55rem;
        margin: .75rem 0 1rem;
    }
    .science-legend-item {
        display: flex;
        align-items: center;
        gap: .55rem;
        min-width: 0;
        padding: .65rem .7rem;
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 12px;
        background: #0f1720;
        color: #f8fafc !important;
        font-size: .72rem;
        font-weight: 900;
        line-height: 1.2;
    }
    .science-legend-dot {
        width: 10px;
        height: 10px;
        flex: 0 0 10px;
        border-radius: 50%;
        box-shadow: 0 0 0 3px rgba(255,255,255,.07);
    }
    .science-camera-toolbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: .7rem;
        margin: 0 0 .7rem;
        padding: .7rem;
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 14px;
        background: #0f1720;
    }
    .science-camera-status {
        color: #f8fafc !important;
        font-size: .72rem;
        font-weight: 850;
        line-height: 1.35;
    }
    .science-camera-status strong { color: var(--accent) !important; }
    .science-header .science-chip {
        color: #f8fafc !important;
        background: #18222d !important;
        border-color: rgba(255,255,255,.18) !important;
    }
    .science-header .science-chip:first-of-type {
        color: #071008 !important;
        background: var(--accent) !important;
        border-color: var(--accent) !important;
    }
    @media (max-width: 640px) {
        .science-legend { grid-template-columns: 1fr; }
        .science-legend-item { min-height: 42px; }
        .science-camera-toolbar { align-items: stretch; flex-direction: column; }
    }

    /* WebRTC control strip and generated camera preview. */
    iframe[src*="streamlit_webrtc"] {
        width: 100% !important;
        min-width: 100% !important;
        height: 128px !important;
        min-height: 128px !important;
        aspect-ratio: auto !important;
        display: block;
        border: 0;
        border-radius: 14px;
        background: #080d13;
    }
    [data-testid="column"]:has(iframe[src*="streamlit_webrtc"]) {
        min-width: 0 !important;
        width: 100% !important;
    }
    [data-testid="stImage"] img {
        border-radius: 16px;
    }

    /* Fullscreen keeps the presentation surface dark and clean. */
    [data-testid="stAppViewContainer"]:fullscreen,
    [data-testid="stAppViewContainer"]:fullscreen::backdrop,
    body:has([data-testid="stAppViewContainer"]:fullscreen) {
        background: var(--bg) !important;
    }

    /* Tablet layout. */
    @media (max-width: 900px) {
        [data-testid="stAppViewContainer"] > .main {
            padding: .9rem .75rem 1.5rem;
        }
        .app-header {
            align-items: flex-start;
            border-radius: 18px;
        }
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
        .science-video-panel, .science-result-panel {
            min-height: auto;
        }
    }

    /* Phone layout: no cramped two-column controls, no horizontal page overflow. */
    @media (max-width: 640px) {
        [data-testid="stAppViewContainer"] > .main {
            padding: .65rem .55rem 1.25rem;
        }
        .app-header {
            flex-direction: column;
            gap: .8rem;
            padding: 1rem;
            border-radius: 16px;
        }
        .app-title { font-size: 1.35rem; }
        .app-subtitle {
            font-size: .65rem;
            line-height: 1.45;
        }
        .mode-badge {
            align-self: flex-start;
            padding: .5rem .7rem;
            font-size: .62rem;
        }
        [data-testid="stRadio"] {
            margin: .75rem 0 .65rem;
        }
        [data-testid="stRadio"] [role="radiogroup"] {
            border-radius: 13px;
            gap: .35rem;
            padding: .3rem;
        }
        [data-testid="stRadio"] [role="radiogroup"] label {
            flex: 0 0 auto;
            min-height: 42px;
            padding: .58rem .78rem !important;
        }
        [data-testid="stRadio"] [role="radiogroup"] label p,
        [data-testid="stRadio"] [role="radiogroup"] label span {
            font-size: .68rem !important;
        }
        .panel {
            padding: .9rem;
            border-radius: 15px;
        }
        .decision {
            min-height: 155px;
            padding: 1.2rem .8rem;
            border-radius: 15px;
        }
        .decision h2 { font-size: 1.65rem; }
        .guidance { font-size: .82rem; }
        .stButton > button,
        [data-testid="stDownloadButton"] > button {
            min-height: 48px;
            width: 100%;
        }
        [data-testid="stFileUploaderDropzone"] {
            min-height: 125px;
        }
        .science-header {
            padding: .65rem;
            border-radius: 15px;
        }
        .science-title { flex-basis: 100%; }
        .science-chip {
            font-size: .58rem;
            padding: .42rem .55rem;
        }
        .science-decision {
            min-height: 150px;
            border-radius: 15px;
        }
        .science-decision h2 { font-size: 1.55rem; }
        iframe[src*="streamlit_webrtc"] {
            height: 112px !important;
            min-height: 112px !important;
            border-radius: 12px;
        }
        .science-footer {
            font-size: .68rem;
        }
        [data-testid="stImage"] img {
            max-height: 52vh;
            object-fit: contain;
        }
    }

    /* Small-screen accessibility and touch comfort. */
    @media (pointer: coarse) {
        button, [role="radio"], input, select, textarea {
            touch-action: manipulation;
        }
    }
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
if "science_sound_enabled" not in st.session_state:
    st.session_state["science_sound_enabled"] = True
if "science_reference_signature" not in st.session_state:
    st.session_state["science_reference_signature"] = None
if "science_reference_result" not in st.session_state:
    st.session_state["science_reference_result"] = None
if "science_last_beep_result" not in st.session_state:
    st.session_state["science_last_beep_result"] = None
if "science_beep_id" not in st.session_state:
    st.session_state["science_beep_id"] = 0

api_url = st.session_state["api_url"]
mobile_browser = _is_mobile_browser()
requested_science_mode = st.sidebar.toggle("SCIENCE FAIR MODE", value=False)

if requested_science_mode and mobile_browser:
    st.sidebar.warning(
        "🎪 SCIENCE FAIR MODE is designed for desktop/laptop browsers. "
        "Please open this app on a desktop or laptop for the Science Fair presentation."
    )
    science_mode = False
else:
    science_mode = requested_science_mode

st.sidebar.markdown("### SYSTEM STATUS")
st.sidebar.caption(f"API: {api_url}")

header_mode = "SCIENCE FAIR MODE" if science_mode else "ONLINE MODE"
st.markdown(
    """
    <div class="app-header">
      <div>
        <p class="app-title">AI WASTE DOCTOR</p>
        <p class="app-subtitle">INTELLIGENT WASTE CLASSIFICATION SYSTEM · RECYCLE · SORT · CLEANER PLANET</p>
      </div>
      <div class="mode-badge">__MODE__</div>
    </div>
    """.replace("__MODE__", header_mode),
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




def _scan_zone_gray(frame):
    """Normalize the classifier scan zone for local scene-change detection."""
    height, width = frame.shape[:2]
    box_size = max(32, int(min(width, height) * 0.60))
    left = max(0, (width - box_size) // 2)
    top = max(0, (height - box_size) // 2)
    crop = frame[top:top + box_size, left:left + box_size]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _scene_change_score(reference_frame, current_frame):
    """Detect replacement/removal of an object; never chooses its waste class."""
    if reference_frame is None or current_frame is None:
        return 0.0
    try:
        reference = _scan_zone_gray(reference_frame)
        current = _scan_zone_gray(current_frame)
        pixel_delta = float(cv2.absdiff(reference, current).mean())
        ref_edges = cv2.Canny(reference, 60, 140)
        cur_edges = cv2.Canny(current, 60, 140)
        edge_delta = float(
            cv2.absdiff(ref_edges, cur_edges).mean() / 255.0 * 100.0
        )
        return pixel_delta + 0.20 * edge_delta
    except Exception:
        return 0.0


def _scan_zone_gray(frame):
    """Normalize the classifier scan zone for local scene-change detection."""
    height, width = frame.shape[:2]
    box_size = max(32, int(min(width, height) * 0.60))
    left = max(0, (width - box_size) // 2)
    top = max(0, (height - box_size) // 2)
    crop = frame[top:top + box_size, left:left + box_size]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _scene_change_score(reference_frame, current_frame):
    """Detect replacement/removal of an object; never chooses its waste class."""
    if reference_frame is None or current_frame is None:
        return 0.0
    try:
        reference = _scan_zone_gray(reference_frame)
        current = _scan_zone_gray(current_frame)
        pixel_delta = float(cv2.absdiff(reference, current).mean())
        ref_edges = cv2.Canny(reference, 60, 140)
        cur_edges = cv2.Canny(current, 60, 140)
        edge_delta = float(
            cv2.absdiff(ref_edges, cur_edges).mean() / 255.0 * 100.0
        )
        return pixel_delta + 0.20 * edge_delta
    except Exception:
        return 0.0


def render_live_inference(ctx):
    """Run one automatic live-scan cycle; reruns keep the camera alive."""
    processor = ctx.video_processor if ctx is not None else None
    if processor is None:
        st.info("Start the camera once to begin automatic live detection.")
        return

    live_state = st.session_state["live_scan_state"]
    frame, captured_at = processor.latest_frame()
    width, height, camera_fps = processor.frame_info()

    result_slot = st.empty()
    status_slot = st.empty()
    meta_slot = st.empty()
    preview_slot = st.empty()

    if width and height:
        meta_slot.caption(
            f"CAMERA: {width}×{height} · {camera_fps:.1f} FPS · "
            + ("FINAL DECISION" if live_state.committed_result is not None else "AUTOMATIC AI SCANNING")
        )

    if frame is not None:
        preview_slot.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            width="stretch",
            caption="Automatic live scan · keep one object inside the green box",
        )

    # A final result stays locked until Resume Scan is pressed. Predictions at or below 70% are never finalized; the trained model is sampled again until a stable >=70% result is reached.
    if live_state.committed_result is not None:
        final = live_state.committed_result
        confidence = float(final.get("top_confidence", 0.0))
        confidence_text = f"{confidence:.1f}%"

        final_class = str(final.get("top_class", "Unknown"))
        result_slot.markdown(
            """
            <div class="science-decision confident">
                <h2>FINAL DECISION</h2>
                <div style="font-size:2rem;font-weight:800;margin:.4rem 0">
                    __CLASS__
                </div>
                <p>MODEL CONFIDENCE · __CONFIDENCE__</p>
            </div>
            """.replace("__CLASS__", final_class)
            .replace("__CONFIDENCE__", confidence_text),
            unsafe_allow_html=True,
        )

        correction = ""
        if final.get("semantic_guard_applied"):
            hint = final.get("semantic_hint") or {}
            correction = f" · object cross-check: {hint.get('label', 'material')} → {final.get('top_class', 'Unknown')}"
        status_slot.success(
            f"FINAL DECISION · {final.get('top_class', 'Unknown')} · {confidence_text}"
            f"{correction}"
        )

        beep_signature = final.get("top_class", "") + "|" + str(final.get("top_confidence", ""))
        if st.session_state.get("science_last_beep_result") != beep_signature:
            st.session_state["science_beep_id"] += 1
            st.session_state["science_last_beep_result"] = beep_signature
            play_classification_beep(st.session_state["science_beep_id"])

        if st.button(
            "▶ RESUME SCAN",
            key="resume-live-scan",
            type="primary",
            use_container_width=True,
        ):
            live_state.reset()
            st.session_state["live_last_inference"] = 0.0
            st.session_state["live_last_frame_at"] = 0.0
            st.session_state["live_service_checked"] = True
            st.session_state["live_scan_resume_at"] = time.monotonic() + 0.6
            st.session_state["science_last_beep_result"] = None
            st.rerun()

        return

    resume_at = st.session_state.get("live_scan_resume_at", 0.0)
    if time.monotonic() < resume_at:
        status_slot.info("RESUME SCAN · preparing the next object…")
        time.sleep(0.12)
        st.rerun()

    if frame is None:
        status_slot.info(
            "AUTO MODE · waiting for a camera frame. "
            "Place one object inside the green box."
        )
        time.sleep(0.25)
        st.rerun()

    now = time.monotonic()
    last_inference = st.session_state.get("live_last_inference", 0.0)
    last_frame_at = st.session_state.get("live_last_frame_at", 0.0)

    if (
        captured_at <= last_frame_at
        or now - last_inference < LIVE_INFERENCE_INTERVAL
    ):
        status_slot.info("AUTO MODE · detecting the object automatically…")
        time.sleep(0.12)
        st.rerun()

    st.session_state["live_last_inference"] = now
    st.session_state["live_last_frame_at"] = captured_at

    ok, encoded = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    if not ok:
        status_slot.warning("Could not encode the camera frame.")
        time.sleep(0.25)
        st.rerun()

    status_slot.info("AI · analyzing live object with the trained model…")
    try:
        result = classify_image(
            api_url,
            "live-frame.jpg",
            encoded.tobytes(),
            "image/jpeg",
            max_attempts=1,
            check_health=not st.session_state.get("live_service_checked", False),
            predict_timeout=LIVE_PREDICT_TIMEOUT,
        )
        st.session_state["live_service_checked"] = True
        snapshot = live_state.update(result)
        if not snapshot.is_final:
            status_slot.warning(
                f"NOT FINAL · {snapshot.top_class or 'unknown'} at {snapshot.confidence:.1f}% "
                f"(need >70% and {snapshot.consecutive_frames}/3 stable frames). Rechecking…"
            )
        st.rerun()

    except ClassificationError as error:
        status_slot.error(error.user_message)
        with st.expander("Technical details"):
            st.code(error.technical)
        time.sleep(LIVE_ERROR_COOLDOWN)
        st.rerun()

    except Exception as error:
        status_slot.error(
            f"Live classification failed: {type(error).__name__}: {error}"
        )
        time.sleep(LIVE_ERROR_COOLDOWN)
        st.rerun()

@st.cache_data(show_spinner=False)
def _classification_beep_audio():
    """Generate a tiny self-contained WAV notification sound."""
    sample_rate = 44100
    duration = 0.24
    frames = int(sample_rate * duration)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(frames):
            t = i / sample_rate
            # Two-note confirmation chirp with a short fade in/out.
            if t < 0.11:
                frequency = 880.0
            else:
                frequency = 660.0
            envelope = min(1.0, t / 0.012, (duration - t) / 0.035)
            sample = int(0.28 * envelope * math.sin(2.0 * math.pi * frequency * t) * 32767)
            wav.writeframes(sample.to_bytes(2, byteorder="little", signed=True))
    return buffer.getvalue()


def play_classification_beep(event_id):
    """Play the exact science-fair decision beep when a final result appears."""
    if not st.session_state.get("science_sound_enabled", True):
        return

    # This WAV is the 0.24-second 880 Hz -> 660 Hz confirmation beep used by
    # the Science Fair presentation. It is rendered with autoplay immediately
    # when the final decision is displayed.
    st.audio(
        _classification_beep_audio(),
        format="audio/wav",
        autoplay=True,
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


if mobile_browser and requested_science_mode:
    st.warning(
        "📱 **Science Fair Mode is desktop-only.** "
        "Use a desktop or laptop browser for the live Science Fair presentation. "
        "Live Camera mode remains available on mobile."
    )

if science_mode:
    navigation_options = ["🎪  SCIENCE FAIR"]
    st.session_state["active_view"] = "🎪  SCIENCE FAIR"
else:
    navigation_options = ["📁  UPLOAD IMAGE", "📹  LIVE CAMERA", "🎪  SCIENCE FAIR", "💡  HOW IT WORKS", "⚙️  SETTINGS"]

active_view = st.radio(
    "Application view",
    navigation_options,
    horizontal=True,
    label_visibility="collapsed",
    key="active_view",
)

if active_view == "📁  UPLOAD IMAGE":
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

if active_view == "📹  LIVE CAMERA":
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

if active_view == "🎪  SCIENCE FAIR":
    st.markdown(
        '<div class="science-header"><p class="science-title">AI WASTE DOCTOR</p>'
        '<span class="science-chip">AUTO → CLASSIFY → SORT</span>'
        '<span class="science-chip">LIGHT: SIMULATION</span>'
        '<span class="science-chip">AUTO LIVE: ON</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="science-footer">Science Fair Auto Live Mode · camera starts once · AI decides automatically · new objects re-arm automatically</p>', unsafe_allow_html=True)
    st.caption("🔊 Decision alert sound is enabled for final classifications.")
    fullscreen_button_html = """
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
    """
    st.iframe(fullscreen_button_html, height=55, width=240)

    left, right = st.columns([1.08, .92], gap="medium")
    with left:
        st.markdown('<p class="science-panel-title">LIVE CAMERA FEED</p>', unsafe_allow_html=True)
        st.markdown(
            '<div class="science-camera-toolbar"><div class="science-camera-status">ACTIVE: <strong>LIVE CAMERA</strong> · MOBILE READY</div></div>',
            unsafe_allow_html=True,
        )
        try:
            rtc_configuration, rtc_mode, rtc_error = get_rtc_configuration()
            if rtc_error:
                st.error("TURN configuration failed. " + rtc_error)
            elif rtc_mode == "Open Relay TURN/TCP":
                st.caption("NETWORK: TURN relay over TCP · one-way camera transport")
            else:
                st.caption("NETWORK: Cloudflare TURN/TCP relay · one-way camera transport")

            ctx = webrtc_streamer(
                key="science-fair-camera-v12-auto-live",
                mode=WebRtcMode.SENDONLY,
                rtc_configuration=rtc_configuration,
                video_processor_factory=LiveVideoProcessor,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640, "min": 320},
                        "height": {"ideal": 480, "min": 240},
                        "frameRate": {"ideal": 15, "max": 20},
                        # "ideal" requests the selected mobile camera without
                        # rejecting devices that expose different constraints.
                        "facingMode": {"ideal": "environment"},
                    },
                    "audio": False,
                },
                sendback_audio=False,
                media_toggle_controls=False,
                async_processing=True,
            )

            st.markdown(
                '<div class="science-camera-toolbar"><div class="science-camera-status">ACTIVE: <strong>REAR CAMERA</strong> · AUTO LIVE DETECTION</div></div>',
                unsafe_allow_html=True,
            )

            if ctx is None or not ctx.state.playing:
                st.caption("Press START and allow camera permission. The rear camera is selected first.")
        except Exception as error:
            ctx = None
            st.warning("Live browser video is unavailable in this session. Use the reference image option below.")
            st.caption(f"WebRTC status: {type(error).__name__}")

    with right:
        st.markdown('<p class="science-panel-title">REAL-TIME AI CLASSIFICATION</p>', unsafe_allow_html=True)
        reference_result = st.session_state.get("science_reference_result")
        if reference_result is not None:
            st.markdown('<div class="science-status">REFERENCE IMAGE RESULT · AUTOMATIC CLASSIFICATION</div>', unsafe_allow_html=True)
            render_prediction(reference_result)
        if ctx is None or not ctx.state.playing:
            st.markdown('<div class="science-decision uncertain"><h2>SCANNING OBJECT</h2><p>START CAMERA TO BEGIN</p></div>', unsafe_allow_html=True)
            st.caption("Press START once and allow camera access. Then place an object in the green box; classification is automatic.")
        else:
            render_live_inference(ctx)

if active_view == navigation_options[3]:
    st.markdown('<p class="panel-label">HOW IT WORKS</p>', unsafe_allow_html=True)
    steps = [
        ("01", "CAPTURE", "Upload an image or take a photo of one waste object."),
        ("02", "PREPROCESS", "The service resizes and normalizes the image for the trained model."),
        ("03", "CLASSIFY", "The AI compares the object with Recyclable, Dry Waste, and Wet Waste classes."),
        ("04", "SORT", "Use the trained model decision and disposal guidance for the correct bin."),
        ("05", "ACCURACY GATE", "Live mode never finalizes a prediction at 70% confidence or below. It keeps rechecking the trained model until the same class reaches above 70% for 3 consecutive frames."),
        ("06", "RESUME SCAN", "Press Resume Scan after a final result to clear it and automatically begin detecting the next object without a hard refresh."),
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