#!/usr/bin/env python3

from __future__ import annotations

import html
import json
import sys
import tempfile
import textwrap
import time

from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd
import plotly.graph_objects as go
import streamlit as st


from robotreplay.ingestion.upload import (
    UploadSecurityError,
    safe_extract_rosbag_zip,
)
from robotreplay.ingestion.validation import BagValidationError
from robotreplay.services.analyzer import AnalysisError, analyze_bag


RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"


SAMPLE_MISSIONS = {
    "Healthy Navigation": {
        "mission": "healthy_003",
        "note": "No supported failure signature identified.",
    },
    "Localization Jump": {
        "mission": "localization_jump_001",
        "note": "Abrupt localization discontinuity with correlated map-to-odom correction.",
    },
    "LiDAR Dropout": {
        "mission": "lidar_dropout_001",
        "note": "LiDAR scan stream becomes stale during navigation.",
    },
    "TF Delay / Discontinuity": {
        "mission": "tf_delay_001",
        "note": "map-to-odom transform stops updating and later recovers.",
    },
    "Wheel / Encoder Direction Mismatch": {
        "mission": "wheel_mismatch_001",
        "note": "Wheel feedback contradicts expected differential-drive response.",
    },
    "LiDAR Stream Integrity Failure": {
        "mission": "Test_3-E1",
        "note": "External public development recording showing corrupted LiDAR stream integrity.",
    },
}


st.set_page_config(
    page_title="RobotReplay",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
:root {
    --bg: #F4F3EF;
    --surface: rgba(255,255,255,0.90);
    --surface-solid: #FFFFFF;
    --ink: #152238;
    --muted: #6C788B;
    --blue: #286BF1;
    --blue-soft: #EAF0FF;
    --cyan: #14B8C4;
    --coral: #FF6577;
    --coral-soft: #FFF0F2;
    --green: #20B879;
    --green-soft: #E9F9F2;
    --amber: #F5A524;
    --amber-soft: #FFF7E7;
    --navy: #101C31;
    --border: rgba(25,48,78,0.10);
    --border-strong: rgba(25,48,78,0.16);
    --shadow: 0 14px 40px rgba(42,57,83,0.08);
    --shadow-heavy: 0 22px 60px rgba(28,45,76,0.14);
}

html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    color: var(--ink);
    background:
        radial-gradient(circle at 92% -2%, rgba(40,107,241,0.16), transparent 30%),
        radial-gradient(circle at 5% 12%, rgba(20,184,196,0.13), transparent 25%),
        radial-gradient(circle at 58% 62%, rgba(116,87,234,0.055), transparent 34%),
        linear-gradient(180deg, #F8FAFE 0%, var(--bg) 42%, #F5F5F1 100%);
}

[data-testid="stMainBlockContainer"] {
    max-width: 1420px !important;
    padding-top: 1.4rem !important;
    padding-bottom: 4rem !important;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

a { color: var(--blue); }

.rr-navbar {
    position: relative;
    overflow: hidden;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1.5rem;
    padding: 1.05rem 1.25rem;
    margin-bottom: 1.15rem;
    border: 1px solid rgba(40,107,241,0.12);
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(246,249,255,0.86));
    box-shadow: var(--shadow-heavy);
    backdrop-filter: blur(20px);
}

.rr-navbar::after {
    content: "";
    position: absolute;
    inset: auto 0 0 0;
    height: 3px;
    background: linear-gradient(90deg, #286BF1, #14B8C4, #7457EA);
    opacity: 0.92;
}

.rr-brand-wrap { display: flex; align-items: center; gap: 0.95rem; }

.rr-logo-shell {
    width: 64px;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 20px;
    background: linear-gradient(145deg, #101C31, #1D3A72 58%, #286BF1);
    box-shadow: 0 14px 32px rgba(40,107,241,0.27), inset 0 0 0 1px rgba(255,255,255,0.10);
}

.rr-logo-shell svg { width: 50px; height: 50px; display: block; }

.rr-brand {
    font-size: 1.82rem;
    line-height: 1;
    font-weight: 930;
    letter-spacing: -0.055em;
}

.rr-brand-accent {
    background: linear-gradient(90deg, #286BF1, #14B8C4);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.rr-tagline { margin-top: 0.34rem; color: var(--muted); font-size: 0.78rem; font-weight: 560; }
.rr-brand-note { margin-top: 0.18rem; color: #8793A6; font-size: 0.61rem; letter-spacing: 0.06em; text-transform: uppercase; font-weight: 760; }
.rr-pills { display: flex; justify-content: flex-end; gap: 0.42rem; flex-wrap: wrap; }
.rr-pill {
    padding: 0.37rem 0.65rem;
    border-radius: 999px;
    color: #536176;
    background: rgba(255,255,255,0.86);
    border: 1px solid var(--border);
    font-size: 0.67rem;
    font-weight: 750;
}
.rr-live-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    margin-right: 0.35rem;
    border-radius: 999px;
    background: var(--green);
    box-shadow: 0 0 0 4px rgba(32,184,121,0.10);
}

button[data-baseweb="tab"] { font-weight: 760; color: #68768A; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--blue) !important; }
div[data-baseweb="tab-highlight"] { background-color: var(--blue) !important; }

.rr-upload-hero {
    display: grid;
    grid-template-columns: 1.25fr 0.75fr;
    gap: 1rem;
    margin: 0.55rem 0 1rem;
}

.rr-upload-copy-card {
    padding: 1.45rem;
    border: 1px solid var(--border);
    border-radius: 24px;
    background:
        radial-gradient(circle at 96% 8%, rgba(20,184,196,0.10), transparent 25%),
        linear-gradient(135deg, rgba(255,255,255,0.98), rgba(244,248,255,0.94));
    box-shadow: var(--shadow-heavy);
}

.rr-eyebrow {
    display: inline-flex;
    padding: 0.28rem 0.55rem;
    margin-bottom: 0.75rem;
    border-radius: 999px;
    color: var(--blue);
    background: var(--blue-soft);
    font-size: 0.64rem;
    font-weight: 840;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

.rr-upload-heading {
    max-width: 780px;
    font-size: 1.8rem;
    line-height: 1.15;
    font-weight: 880;
    letter-spacing: -0.035em;
}

.rr-upload-desc {
    max-width: 820px;
    margin-top: 0.55rem;
    color: var(--muted);
    line-height: 1.55;
    font-size: 0.86rem;
}

.rr-flow-card {
    padding: 1.15rem;
    border-radius: 22px;
    color: white;
    background: linear-gradient(145deg, #192B49, #286BF1);
    box-shadow: 0 16px 35px rgba(40,107,241,0.16);
}

.rr-flow-title {
    margin-bottom: 0.75rem;
    font-size: 0.70rem;
    font-weight: 850;
    letter-spacing: 0.08em;
    opacity: 0.76;
    text-transform: uppercase;
}

.rr-flow-item { display: flex; align-items: center; gap: 0.55rem; padding: 0.45rem 0; font-size: 0.76rem; }
.rr-flow-number {
    min-width: 1.55rem;
    width: 1.55rem;
    height: 1.55rem;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    background: rgba(255,255,255,0.14);
    font-size: 0.66rem;
    font-weight: 850;
}

.rr-trust-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.55rem;
    margin-bottom: 0.9rem;
}

.rr-trust-item {
    padding: 0.68rem 0.75rem;
    text-align: center;
    border: 1px solid var(--border);
    border-radius: 13px;
    color: #66748A;
    background: rgba(255,255,255,0.70);
    font-size: 0.68rem;
    font-weight: 670;
}

.rr-trust-icon { color: var(--green); margin-right: 0.27rem; font-weight: 900; }

[data-testid="stFileUploader"] {
    padding: 0.75rem;
    border: 2px dashed rgba(40,107,241,0.28);
    border-radius: 18px;
    background: rgba(255,255,255,0.77);
}
[data-testid="stFileUploader"]:hover { border-color: rgba(40,107,241,0.55); }

div.stButton > button {
    width: 100%;
    padding: 0.76rem 1.1rem;
    border: none;
    border-radius: 13px;
    color: white;
    background: linear-gradient(135deg, #286BF1, #5B56E8);
    box-shadow: 0 10px 22px rgba(40,107,241,0.15);
    font-weight: 780;
}

div.stDownloadButton > button {
    width: 100%;
    border: 1px solid var(--border-strong);
    border-radius: 12px;
    color: var(--ink);
    background: white;
}

.rr-result-shell {
    position: relative;
    overflow: hidden;
    padding: 1.25rem 1.25rem 1.25rem 1.45rem;
    margin: 0.45rem 0 0.95rem;
    border: 1px solid var(--border);
    border-radius: 22px;
    background: rgba(255,255,255,0.88);
    box-shadow: var(--shadow);
}
.rr-result-shell::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 5px;
    background: var(--amber);
}
.rr-result-shell.result-healthy::before { background: var(--green); }
.rr-result-shell.result-anomalous::before { background: var(--coral); }
.rr-result-shell.result-unknown::before { background: var(--amber); }
.rr-result-shell.result-healthy {
    background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(242,253,248,0.88));
}
.rr-result-shell.result-anomalous {
    background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(255,246,248,0.90));
}

.rr-result-top { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
.rr-result-label {
    color: var(--muted);
    font-size: 0.64rem;
    font-weight: 850;
    letter-spacing: 0.10em;
    text-transform: uppercase;
}
.rr-result-cause {
    margin-top: 0.25rem;
    font-size: 1.72rem;
    line-height: 1.1;
    font-weight: 880;
    letter-spacing: -0.035em;
}
.rr-result-meta { margin-top: 0.5rem; color: var(--muted); font-size: 0.72rem; }
.rr-status-badge {
    padding: 0.47rem 0.68rem;
    border-radius: 999px;
    font-size: 0.67rem;
    font-weight: 850;
    letter-spacing: 0.055em;
}
.rr-status-anomalous { color: #C53E53; background: var(--coral-soft); border: 1px solid rgba(255,101,119,0.24); }
.rr-status-healthy { color: #15825B; background: var(--green-soft); border: 1px solid rgba(32,184,121,0.22); }
.rr-status-unknown { color: #B27618; background: var(--amber-soft); border: 1px solid rgba(245,165,36,0.23); }

.rr-crosscheck {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.82rem 0.95rem;
    margin-bottom: 0.9rem;
    border-radius: 14px;
    font-size: 0.74rem;
}
.rr-crosscheck-good { color: #176E52; background: var(--green-soft); border: 1px solid rgba(32,184,121,0.22); }
.rr-crosscheck-warn { color: #8A5E17; background: var(--amber-soft); border: 1px solid rgba(245,165,36,0.23); }
.rr-crosscheck-bad { color: #A3394A; background: var(--coral-soft); border: 1px solid rgba(255,101,119,0.25); }
.rr-crosscheck-title { font-weight: 850; }

.rr-metric-card {
    min-height: 100px;
    padding: 0.9rem;
    border: 1px solid var(--border);
    border-radius: 16px;
    background: rgba(255,255,255,0.88);
    box-shadow: 0 7px 20px rgba(42,57,83,0.045);
}
.rr-metric-label {
    color: var(--muted);
    font-size: 0.64rem;
    font-weight: 820;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}
.rr-metric-value { margin-top: 0.30rem; font-size: 1.15rem; font-weight: 860; }
.rr-metric-sub { margin-top: 0.20rem; color: #8994A5; font-size: 0.69rem; }

.rr-section-card {
    padding: 1rem;
    border: 1px solid var(--border);
    border-radius: 18px;
    background: rgba(255,255,255,0.84);
}
.rr-section-kicker {
    color: var(--blue);
    font-size: 0.64rem;
    font-weight: 850;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.rr-section-title { margin-top: 0.25rem; font-size: 1.03rem; font-weight: 850; }
.rr-section-copy { margin-top: 0.55rem; color: #4B596E; font-size: 0.82rem; line-height: 1.6; }

.rr-evidence-box {
    padding: 0.75rem 0.85rem;
    margin-bottom: 0.5rem;
    border: 1px solid var(--border);
    border-left: 4px solid var(--cyan);
    border-radius: 12px;
    color: #435168;
    background: white;
    font-size: 0.80rem;
}

.rr-action {
    display: flex;
    align-items: flex-start;
    gap: 0.70rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid var(--border);
}
.rr-action-index {
    min-width: 1.75rem;
    width: 1.75rem;
    height: 1.75rem;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 9px;
    color: var(--blue);
    background: var(--blue-soft);
    font-size: 0.69rem;
    font-weight: 850;
}

.rr-sequence {
    display: grid;
    gap: 0.58rem;
}
.rr-sequence-item {
    display: grid;
    grid-template-columns: 34px minmax(0, 1fr);
    gap: 0.68rem;
    align-items: start;
}
.rr-sequence-index {
    width: 31px;
    height: 31px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    color: var(--blue);
    background: var(--blue-soft);
    border: 1px solid rgba(40,107,241,0.17);
    font-size: 0.69rem;
    font-weight: 900;
    box-shadow: 0 5px 12px rgba(40,107,241,0.07);
}
.rr-sequence-item.first .rr-sequence-index {
    color: #C53E53;
    background: var(--coral-soft);
    border-color: rgba(255,101,119,0.24);
}
.rr-sequence-card {
    min-height: 31px;
    padding: 0.68rem 0.78rem;
    border: 1px solid var(--border);
    border-radius: 13px;
    color: #435168;
    background: rgba(255,255,255,0.93);
    box-shadow: 0 6px 18px rgba(42,57,83,0.035);
    font-size: 0.80rem;
    line-height: 1.48;
}
.rr-sequence-item.first .rr-sequence-card {
    border-color: rgba(255,101,119,0.22);
    background: linear-gradient(135deg, #FFF7F8, #FFFFFF);
    box-shadow: 0 7px 20px rgba(255,101,119,0.055);
}
.rr-sequence-label {
    margin-bottom: 0.22rem;
    color: #C53E53;
    font-size: 0.63rem;
    font-weight: 900;
    letter-spacing: 0.065em;
    text-transform: uppercase;
}
.rr-sequence-time {
    display: inline-flex;
    align-items: center;
    padding: 0.20rem 0.42rem;
    margin-right: 0.34rem;
    border-radius: 999px;
    color: #C53E53;
    background: #FFE8EC;
    font-size: 0.65rem;
    font-weight: 900;
}

.rr-signature-row {
    display: grid;
    grid-template-columns: 1fr auto auto;
    gap: 0.75rem;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.76rem;
}
.rr-signature-score { color: var(--blue); font-weight: 850; }
.rr-signature-state { color: var(--muted); font-size: 0.68rem; }


.rr-report-cta {
    position: relative;
    overflow: hidden;
    padding: 1.15rem 1.2rem;
    margin: 0.25rem 0 0.8rem;
    border-radius: 20px;
    color: white;
    background:
        radial-gradient(circle at 92% 10%, rgba(101,243,232,0.20), transparent 28%),
        linear-gradient(135deg, #111F37, #1D3F7A 58%, #286BF1);
    box-shadow: 0 18px 42px rgba(33,66,130,0.20);
}
.rr-report-cta::after {
    content: "PDF";
    position: absolute;
    right: 1rem;
    top: -0.45rem;
    font-size: 4.8rem;
    line-height: 1;
    font-weight: 950;
    letter-spacing: -0.08em;
    color: rgba(255,255,255,0.055);
}
.rr-report-badge {
    display: inline-flex;
    padding: 0.26rem 0.48rem;
    margin-bottom: 0.45rem;
    border-radius: 999px;
    color: #B7FFF8;
    background: rgba(101,243,232,0.10);
    border: 1px solid rgba(101,243,232,0.18);
    font-size: 0.61rem;
    font-weight: 850;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.rr-report-title { font-size: 1.12rem; font-weight: 880; letter-spacing: -0.02em; }
.rr-report-copy { margin-top: 0.3rem; max-width: 760px; color: rgba(255,255,255,0.76); font-size: 0.75rem; line-height: 1.5; }
.rr-report-meta { margin-top: 0.48rem; color: rgba(255,255,255,0.60); font-size: 0.64rem; }

.rr-footer {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    margin-top: 2rem;
    padding: 1rem 0;
    color: #8994A5;
    border-top: 1px solid var(--border);
    font-size: 0.65rem;
}

[data-testid="stMetric"] {
    border: 1px solid var(--border);
    border-radius: 14px;
    background: white;
    padding: 0.75rem 0.85rem;
}

div[data-testid="stAlert"] { border-radius: 13px; }

@media (max-width: 950px) {
    .rr-navbar { flex-direction: column; align-items: flex-start; }
    .rr-pills { justify-content: flex-start; }
    .rr-upload-hero { grid-template-columns: 1fr; }
    .rr-trust-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
""",
    unsafe_allow_html=True,
)


def render_html(markup: str) -> None:
    markup = textwrap.dedent(markup).strip()
    markup = " ".join(
        line.strip()
        for line in markup.splitlines()
        if line.strip()
    )
    st.markdown(markup, unsafe_allow_html=True)


def safe_text(value) -> str:
    return html.escape(str(value))


def format_bytes(value) -> str:
    if value is None:
        return "—"

    value = float(value)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024

    return f"{value:.1f} TB"


def evidence_label(score: float) -> str:
    if score >= 0.85:
        return "HIGH"
    if score >= 0.65:
        return "MODERATE"
    if score > 0:
        return "LOW"
    return "NONE"


def render_header() -> None:
    render_html(
        """
        <div class="rr-navbar">
            <div class="rr-brand-wrap">
                <div class="rr-logo-shell" aria-label="RobotReplay logo">
                    <svg viewBox="0 0 72 72" xmlns="http://www.w3.org/2000/svg" role="img">
                        <defs>
                            <linearGradient id="rrRing" x1="8" y1="8" x2="64" y2="64" gradientUnits="userSpaceOnUse">
                                <stop stop-color="#65F3E8"/>
                                <stop offset="1" stop-color="#6C8CFF"/>
                            </linearGradient>
                        </defs>
                        <path d="M15 35C15 23.4 24.4 14 36 14C45.8 14 54 20.7 56.3 29.8" fill="none" stroke="url(#rrRing)" stroke-width="4.5" stroke-linecap="round"/>
                        <path d="M57 20L58.2 31.8L46.6 29.6" fill="none" stroke="#65F3E8" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M57 37C57 48.6 47.6 58 36 58C26.1 58 17.8 51.2 15.6 41.9" fill="none" stroke="url(#rrRing)" stroke-width="4.5" stroke-linecap="round" opacity="0.92"/>
                        <rect x="23" y="25" width="26" height="22" rx="7" fill="rgba(255,255,255,0.08)" stroke="white" stroke-width="2.8"/>
                        <circle cx="31" cy="35" r="2.6" fill="#65F3E8"/>
                        <circle cx="41" cy="35" r="2.6" fill="#65F3E8"/>
                        <path d="M31 41H41" stroke="white" stroke-width="2.6" stroke-linecap="round"/>
                        <path d="M36 25V20" stroke="white" stroke-width="2.4" stroke-linecap="round"/>
                        <circle cx="36" cy="18" r="2" fill="#65F3E8"/>
                    </svg>
                </div>
                <div>
                    <div class="rr-brand">
                        ROBOT<span class="rr-brand-accent">REPLAY</span>
                    </div>
                    <div class="rr-tagline">Incident investigation for autonomous robots.</div>
                    <div class="rr-brand-note">Black-box replay · causal evidence · engineering actions</div>
                </div>
            </div>
            <div class="rr-pills">
                <div class="rr-pill"><span class="rr-live-dot"></span>Local analysis</div>
                <div class="rr-pill">ROS 2</div>
                <div class="rr-pill">SQLite3 + MCAP</div>
                <div class="rr-pill">Explainable evidence</div>
            </div>
        </div>
        """
    )


def render_metric_card(label: str, value: str, subtext: str) -> None:
    render_html(
        f"""
        <div class="rr-metric-card">
            <div class="rr-metric-label">{safe_text(label)}</div>
            <div class="rr-metric-value">{safe_text(value)}</div>
            <div class="rr-metric-sub">{safe_text(subtext)}</div>
        </div>
        """
    )


def render_crosscheck(report: dict) -> None:
    agreement = report.get("intelligence_agreement", {})

    if not agreement.get("available"):
        style = "rr-crosscheck-warn"
        title = "Evidence-based diagnosis available"
        summary = (
            "Additional statistical cross-checks were not applicable "
            "to the telemetry available in this recording."
        )
    else:
        state = agreement.get("state", "UNKNOWN")

        if state == "AGREEMENT":
            style = "rr-crosscheck-good"
            title = "Independent cross-checks support this diagnosis"
            summary = (
                "Multiple diagnostic paths independently support the same incident interpretation."
            )
        elif state == "DISAGREEMENT":
            style = "rr-crosscheck-bad"
            title = "Diagnostic cross-checks disagree"
            summary = (
                "RobotReplay retains the evidence-based diagnosis and surfaces the disagreement for review."
            )
        else:
            style = "rr-crosscheck-warn"
            title = "Diagnostic cross-check is partial"
            summary = (
                "Evidence reasoning produced a result, but not every secondary cross-check resolved the incident."
            )

    render_html(
        f"""
        <div class="rr-crosscheck {style}">
            <div>
                <span class="rr-crosscheck-title">{safe_text(title)}</span>
                &nbsp;·&nbsp; {safe_text(summary)}
            </div>
        </div>
        """
    )


def incident_summary(report: dict) -> str:
    status = report.get("mission_status", "UNKNOWN")
    cause = report.get("root_cause", "Unknown")

    summaries = {
        "Localization Jump": (
            "RobotReplay found an abrupt localization change with a correlated "
            "map-to-odom correction. The navigation stack therefore received a "
            "suddenly changed robot pose rather than a gradual motion update."
        ),
        "LiDAR Dropout": (
            "The LiDAR stream stopped refreshing for substantially longer than "
            "the healthy baseline. Navigation was left with stale obstacle data "
            "until the sensor stream recovered."
        ),
        "LiDAR Stream Integrity Failure": (
            "LiDAR messages continued to arrive, but the stream itself became "
            "unreliable: timestamp ordering degraded, scan payloads repeated, "
            "and usable range content collapsed before downstream navigation "
            "state became stale."
        ),
        "TF Delay / Discontinuity": (
            "The map-to-odom transform stopped updating while lower-level robot "
            "state remained comparatively fresh. This localized the incident to "
            "the navigation transform path rather than a complete telemetry loss."
        ),
        "Wheel / Encoder Direction Mismatch": (
            "Wheel feedback contradicted the direction expected from the velocity "
            "command. The mismatch persisted across synchronized windows, indicating "
            "a motor, encoder-sign, or wheel-configuration inconsistency."
        ),
    }

    if cause in summaries:
        return summaries[cause]

    if status == "HEALTHY":
        return (
            "RobotReplay did not establish any of the currently supported failure "
            "signatures in this recording. This is a result for the available "
            "diagnostic coverage, not a guarantee that every possible robot fault is absent."
        )

    if cause == "No supported fault detected":
        return (
            "Abnormal behavior was detected, but the available evidence did not "
            "establish a supported root-cause signature. RobotReplay therefore "
            "leaves the incident unresolved rather than forcing it into a known class."
        )

    return (
        "RobotReplay reconstructed the recording and produced an evidence-based "
        "incident result from the telemetry available."
    )


def render_what_happened(report: dict) -> None:
    render_html(
        f"""
        <div class="rr-section-card">
            <div class="rr-section-kicker">What happened</div>
            <div class="rr-section-title">Incident interpretation</div>
            <div class="rr-section-copy">{safe_text(incident_summary(report))}</div>
        </div>
        """
    )


def top_hypothesis(report: dict) -> dict:
    root = report.get("root_cause")

    for item in report.get("ranked_hypotheses", []):
        if item.get("name") == root:
            return item

    hypotheses = report.get("ranked_hypotheses", [])
    return hypotheses[0] if hypotheses else {}


def flatten_evidence(value, prefix: str = "") -> list[dict]:
    rows = []

    if isinstance(value, dict):
        for key, child in value.items():
            label = f"{prefix} {key}".strip()
            rows.extend(flatten_evidence(child, label))
        return rows

    if isinstance(value, (list, tuple)):
        if value and all(not isinstance(item, (dict, list, tuple)) for item in value):
            rows.append({"Signal": prefix.replace("_", " ").title(), "Value": ", ".join(map(str, value))})
        return rows

    rows.append({
        "Signal": prefix.replace("_", " ").title(),
        "Value": value,
    })
    return rows


def add_trace(
    fig: go.Figure,
    features: pd.DataFrame,
    column: str,
    name: str,
    color: str,
) -> None:
    if column not in features.columns or "time" not in features.columns:
        return

    fig.add_trace(
        go.Scatter(
            x=features["time"],
            y=features[column],
            mode="lines",
            name=name,
            line=dict(width=2.2, color=color),
        )
    )


def build_telemetry_chart(report: dict, features: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    cause = report.get("root_cause", "")

    if cause == "Localization Jump":
        add_trace(fig, features, "amcl_position_step", "AMCL position step", "#286BF1")
        add_trace(fig, features, "tf_map_odom_position_step", "map→odom position step", "#FF6577")
        y_title = "Position change (m)"

    elif cause == "LiDAR Dropout":
        add_trace(fig, features, "scan_age", "LiDAR scan age", "#286BF1")
        y_title = "Scan age (s)"

    elif cause == "LiDAR Stream Integrity Failure":
        add_trace(fig, features, "scan_header_nonmonotonic_fraction_1s", "Non-monotonic timestamps", "#FF6577")
        add_trace(fig, features, "scan_payload_repeat_fraction_1s", "Repeated scans", "#286BF1")
        add_trace(fig, features, "scan_low_quality_fraction_1s", "Degraded scans", "#14B8C4")
        y_title = "Fraction of scans"

    elif cause == "TF Delay / Discontinuity":
        add_trace(fig, features, "tf_map_odom_age", "map→odom age", "#FF6577")
        add_trace(fig, features, "tf_odom_base_age", "odom→base age", "#14B8C4")
        y_title = "Transform age (s)"

    elif cause == "Wheel / Encoder Direction Mismatch":
        add_trace(fig, features, "left_wheel_vel", "Left wheel", "#286BF1")
        add_trace(fig, features, "right_wheel_vel", "Right wheel", "#FF6577")
        y_title = "Wheel velocity (rad/s)"

    else:
        add_trace(fig, features, "cmd_vx", "Commanded velocity", "#286BF1")
        add_trace(fig, features, "odom_vx", "Observed velocity", "#14B8C4")
        y_title = "Velocity (m/s)"

    event = report.get("first_abnormal_time_s")

    if event is not None:
        event_value = float(event)
        visual_half_width = 0.25

        fig.add_vrect(
            x0=event_value - visual_half_width,
            x1=event_value + visual_half_width,
            fillcolor="rgba(255,101,119,0.09)",
            line_width=0,
            layer="below",
        )
        fig.add_vline(
            x=event_value,
            line_width=2,
            line_dash="dash",
            line_color="#FF6577",
            annotation_text="First evidence",
            annotation_position="top right",
        )

    fig.update_layout(
        height=390,
        margin=dict(l=15, r=15, t=40, b=15),
        xaxis=dict(
            title="Mission time (s)",
            showgrid=True,
            gridcolor="rgba(53,73,104,0.08)",
            zeroline=False,
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor="rgba(53,73,104,0.08)",
            zeroline=False,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#445268"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return fig


def render_supporting_evidence(report: dict) -> None:
    supporting = report.get("supporting_evidence", [])

    if supporting:
        for item in supporting:
            render_html(
                f"""
                <div class="rr-evidence-box">{safe_text(item)}</div>
                """
            )
        return

    if report.get("mission_status") == "HEALTHY":
        st.success("No supported failure signature was identified.")
    else:
        st.info("No supporting evidence summary is available for this incident.")


def render_failure_timeline(report: dict) -> None:
    chain = report.get("failure_chain", [])
    event = report.get("first_abnormal_time_s")

    items = []
    step_number = 1

    if event is not None:
        items.append(
            f"""
            <div class="rr-sequence-item first">
                <div class="rr-sequence-index">{step_number}</div>
                <div class="rr-sequence-card">
                    <div class="rr-sequence-label">Incident onset</div>
                    <span class="rr-sequence-time">{float(event):.3f} s</span>
                    First supported diagnostic evidence appears.
                </div>
            </div>
            """
        )
        step_number += 1

    for step in chain:
        items.append(
            f"""
            <div class="rr-sequence-item">
                <div class="rr-sequence-index">{step_number}</div>
                <div class="rr-sequence-card">{safe_text(step)}</div>
            </div>
            """
        )
        step_number += 1

    if not items:
        st.info("No failure propagation was reconstructed.")
        return

    render_html(
        f"""
        <div class="rr-sequence">{''.join(items)}</div>
        """
    )


def render_actions(report: dict) -> None:
    checks = report.get("recommended_checks", [])

    if checks:
        for index, check in enumerate(checks, start=1):
            render_html(
                f"""
                <div class="rr-action">
                    <div class="rr-action-index">{index}</div>
                    <div>{safe_text(check)}</div>
                </div>
                """
            )
        return

    if report.get("mission_status") == "HEALTHY":
        st.info("No corrective action is indicated by the supported signatures.")
    else:
        st.info("No recommended checks were generated for this incident.")


def render_other_signatures(report: dict) -> None:
    rows = []

    for item in report.get("ranked_hypotheses", []):
        name = item.get("name", "Unknown")
        score = float(item.get("score", 0.0))
        available = item.get("available", True)
        state = "checked" if available else "not applicable"

        rows.append(
            f"""
            <div class="rr-signature-row">
                <div>{safe_text(name)}</div>
                <div class="rr-signature-score">{score:.2f}</div>
                <div class="rr-signature-state">{safe_text(state)}</div>
            </div>
            """
        )

    if rows:
        render_html("".join(rows))
    else:
        st.info("No signature-ranking data is available.")


def sanitized_reason(reason: str | None) -> str:
    if not reason:
        return "Not applicable to available telemetry."

    text = str(reason)

    coverage_marker = "only "
    if coverage_marker in text.lower() and "%" in text:
        lower = text.lower()
        start = lower.find(coverage_marker) + len(coverage_marker)
        end = lower.find("%", start)

        if end > start:
            percentage = text[start:end + 1]
            return f"Insufficient telemetry coverage ({percentage})."

    return "Not applicable to available telemetry."



def pdf_safe_text(value) -> str:
    text = str(value if value is not None else "")
    replacements = {
        "→": "->",
        "–": "-",
        "—": "-",
        "×": "x",
        "✓": "",
        "•": "-",
        "≤": "<=",
        "≥": ">=",
        "≈": "~",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _pdf_telemetry_series(report: dict, features: pd.DataFrame):
    cause = report.get("root_cause", "")

    mappings = {
        "Localization Jump": [
            ("amcl_position_step", "AMCL position step"),
            ("tf_map_odom_position_step", "map->odom position step"),
        ],
        "LiDAR Dropout": [
            ("scan_age", "LiDAR scan age"),
        ],
        "LiDAR Stream Integrity Failure": [
            ("scan_header_nonmonotonic_fraction_1s", "Non-monotonic timestamps"),
            ("scan_payload_repeat_fraction_1s", "Repeated scans"),
            ("scan_low_quality_fraction_1s", "Degraded scans"),
        ],
        "TF Delay / Discontinuity": [
            ("tf_map_odom_age", "map->odom age"),
            ("tf_odom_base_age", "odom->base age"),
        ],
        "Wheel / Encoder Direction Mismatch": [
            ("left_wheel_vel", "Left wheel"),
            ("right_wheel_vel", "Right wheel"),
        ],
    }

    selected = mappings.get(
        cause,
        [
            ("cmd_vx", "Commanded velocity"),
            ("odom_vx", "Observed velocity"),
        ],
    )

    series = []

    if "time" not in features.columns:
        return series

    x = pd.to_numeric(features["time"], errors="coerce")

    for column, label in selected:
        if column not in features.columns:
            continue

        y = pd.to_numeric(features[column], errors="coerce")
        valid = x.notna() & y.notna()

        if not valid.any():
            continue

        xs = x[valid].astype(float).tolist()
        ys = y[valid].astype(float).tolist()

        if len(xs) > 220:
            step = max(1, len(xs) // 220)
            xs = xs[::step]
            ys = ys[::step]

        series.append((label, xs, ys))

    return series


def build_pdf_telemetry_drawing(
    report: dict,
    features: pd.DataFrame,
    *,
    compact: bool = False,
):
    from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String
    from reportlab.lib import colors

    width = 500
    height = 142 if compact else 190
    left = 46
    right = 16
    bottom = 28 if compact else 34
    top = 28 if compact else 34
    plot_w = width - left - right
    plot_h = height - bottom - top

    drawing = Drawing(width, height)
    series = _pdf_telemetry_series(report, features)

    if not series:
        drawing.add(
            String(
                12,
                height / 2,
                "Telemetry plot unavailable for this recording.",
                fontName="Helvetica",
                fontSize=9,
                fillColor=colors.HexColor("#6C788B"),
            )
        )
        return drawing

    all_x = [v for _, xs, _ in series for v in xs]
    all_y = [v for _, _, ys in series for v in ys]

    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)

    if abs(x_max - x_min) < 1e-12:
        x_max = x_min + 1.0

    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0

    y_pad = 0.08 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    def sx(value):
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value):
        return bottom + (value - y_min) / (y_max - y_min) * plot_h

    grid_color = colors.HexColor("#E5EAF2")
    axis_color = colors.HexColor("#A9B4C4")

    for i in range(5):
        gx = left + plot_w * i / 4
        gy = bottom + plot_h * i / 4
        drawing.add(Line(gx, bottom, gx, bottom + plot_h, strokeColor=grid_color, strokeWidth=0.6))
        drawing.add(Line(left, gy, left + plot_w, gy, strokeColor=grid_color, strokeWidth=0.6))

    drawing.add(Line(left, bottom, left, bottom + plot_h, strokeColor=axis_color, strokeWidth=0.8))
    drawing.add(Line(left, bottom, left + plot_w, bottom, strokeColor=axis_color, strokeWidth=0.8))

    palette = ["#286BF1", "#FF6577", "#14B8C4"]

    for index, (label, xs, ys) in enumerate(series):
        points = []
        for xv, yv in zip(xs, ys):
            points.extend([sx(xv), sy(yv)])

        drawing.add(
            PolyLine(
                points,
                strokeColor=colors.HexColor(palette[index % len(palette)]),
                strokeWidth=1.5,
            )
        )

        legend_x = left + index * 152
        drawing.add(
            Line(
                legend_x,
                height - 13,
                legend_x + 16,
                height - 13,
                strokeColor=colors.HexColor(palette[index % len(palette)]),
                strokeWidth=2.2,
            )
        )
        drawing.add(
            String(
                legend_x + 20,
                height - 16,
                pdf_safe_text(label)[:28],
                fontName="Helvetica",
                fontSize=6.4 if compact else 6.5,
                fillColor=colors.HexColor("#445268"),
            )
        )

    event = report.get("first_abnormal_time_s")
    if event is not None and x_min <= float(event) <= x_max:
        event_value = float(event)
        visual_half_width = 0.25
        band_start = max(x_min, event_value - visual_half_width)
        band_end = min(x_max, event_value + visual_half_width)
        bx0 = sx(band_start)
        bx1 = sx(band_end)

        drawing.add(
            Rect(
                bx0,
                bottom,
                max(1.5, bx1 - bx0),
                plot_h,
                fillColor=colors.HexColor("#FFF0F2"),
                strokeColor=None,
            )
        )

        # Redraw the series above the subtle onset band.
        for index, (_, xs, ys) in enumerate(series):
            points = []
            for xv, yv in zip(xs, ys):
                points.extend([sx(xv), sy(yv)])
            drawing.add(
                PolyLine(
                    points,
                    strokeColor=colors.HexColor(palette[index % len(palette)]),
                    strokeWidth=1.5,
                )
            )

        ex = sx(event_value)
        drawing.add(
            Line(
                ex,
                bottom,
                ex,
                bottom + plot_h,
                strokeColor=colors.HexColor("#FF6577"),
                strokeWidth=1.2,
                strokeDashArray=[3, 2],
            )
        )
        drawing.add(
            String(
                min(ex + 3, width - 104),
                bottom + plot_h - 10,
                f"First evidence {event_value:.3f}s",
                fontName="Helvetica-Bold",
                fontSize=6.6,
                fillColor=colors.HexColor("#C53E53"),
            )
        )

    drawing.add(
        String(
            left,
            7,
            f"Mission time: {x_min:.1f}s to {x_max:.1f}s",
            fontName="Helvetica",
            fontSize=6.6,
            fillColor=colors.HexColor("#6C788B"),
        )
    )
    drawing.add(
        String(
            width - 118,
            7,
            f"Range: {y_min:.3g} to {y_max:.3g}",
            fontName="Helvetica",
            fontSize=6.6,
            fillColor=colors.HexColor("#6C788B"),
        )
    )

    return drawing

def build_incident_pdf(report: dict, features: pd.DataFrame) -> bytes:
    try:
        from reportlab.graphics.shapes import Circle, Drawing, Line, Rect
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            KeepTogether,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PDF export requires reportlab. Install it with: pip install reportlab==4.4.9"
        ) from exc

    buffer = BytesIO()
    mission = pdf_safe_text(report.get("mission", "mission"))
    status = pdf_safe_text(report.get("mission_status", "UNKNOWN"))
    cause = pdf_safe_text(report.get("root_cause", "Unknown"))
    event = report.get("first_abnormal_time_s")
    evidence_score = float(report.get("evidence_strength", 0.0))
    bag = report.get("bag", {})
    perf = report.get("performance", {})
    is_healthy = status == "HEALTHY"
    is_anomalous = status.startswith("ANOMALOUS")

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=16 * mm,
        title=f"RobotReplay Incident Investigation - {mission}",
        author="RobotReplay",
        subject="Autonomous robot incident investigation",
    )

    styles = getSampleStyleSheet()
    navy = colors.HexColor("#152238")
    blue = colors.HexColor("#286BF1")
    cyan = colors.HexColor("#14B8C4")
    coral = colors.HexColor("#FF6577")
    green = colors.HexColor("#20B879")
    amber = colors.HexColor("#F5A524")
    muted = colors.HexColor("#6C788B")
    pale = colors.HexColor("#F4F7FC")
    border = colors.HexColor("#DEE5EF")
    green_soft = colors.HexColor("#EDF9F4")
    coral_soft = colors.HexColor("#FFF1F3")
    amber_soft = colors.HexColor("#FFF8E9")

    if is_healthy:
        status_accent = green
        status_fill = green_soft
    elif is_anomalous:
        status_accent = coral
        status_fill = coral_soft
    else:
        status_accent = amber
        status_fill = amber_soft

    title_style = ParagraphStyle(
        "RRTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=25,
        textColor=navy,
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "RRSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.3,
        leading=11.5,
        textColor=muted,
        spaceAfter=9,
    )
    heading_style = ParagraphStyle(
        "RRHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        textColor=navy,
        spaceBefore=8,
        spaceAfter=5,
    )
    compact_heading_style = ParagraphStyle(
        "RRCompactHeading",
        parent=heading_style,
        fontSize=10.3,
        leading=12.5,
        spaceBefore=5,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "RRBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.3,
        leading=11.7,
        textColor=colors.HexColor("#46546A"),
        spaceAfter=4,
    )
    compact_body_style = ParagraphStyle(
        "RRCompactBody",
        parent=body_style,
        fontSize=7.5,
        leading=10.2,
        spaceAfter=2.5,
    )
    small_style = ParagraphStyle(
        "RRSmall",
        parent=body_style,
        fontSize=7.0,
        leading=9.5,
        textColor=muted,
    )
    label_style = ParagraphStyle(
        "RRLabel",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=6.3,
        leading=7.5,
        textColor=muted,
        spaceAfter=1,
    )
    metric_style = ParagraphStyle(
        "RRMetric",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=10.2,
        leading=11.5,
        textColor=navy,
        spaceAfter=0,
    )
    status_metric_style = ParagraphStyle(
        "RRStatusMetric",
        parent=metric_style,
        textColor=status_accent,
    )

    def P(text, style=body_style):
        return Paragraph(xml_escape(pdf_safe_text(text)), style)

    def bullet(text, style=body_style):
        return Paragraph(
            "<font color='#286BF1'><b>•</b></font> " + xml_escape(pdf_safe_text(text)),
            style,
        )

    def draw_page(canvas, _doc):
        canvas.saveState()
        page_w, _ = A4
        canvas.setStrokeColor(colors.HexColor("#E5EAF2"))
        canvas.setLineWidth(0.6)
        canvas.line(16 * mm, 11.5 * mm, page_w - 16 * mm, 11.5 * mm)
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(muted)
        canvas.drawString(16 * mm, 7.3 * mm, f"RobotReplay - {mission}")
        canvas.drawRightString(
            page_w - 16 * mm,
            7.3 * mm,
            f"Incident Investigation Report | Page {canvas.getPageNumber()}",
        )
        canvas.restoreState()

    def make_coverage_table(compact=False):
        generic = report.get("generic_ml", {})
        classifier = report.get("fault_classifier", {})
        coverage_rows = [
            [P("ANALYSIS PATH", label_style), P("STATUS", label_style), P("DETAIL", label_style)]
        ]

        if generic.get("available"):
            coverage = generic.get("feature_coverage")
            detail = (
                f"Applicable; {100 * float(coverage):.0f}% feature coverage"
                if coverage is not None
                else "Applicable"
            )
            coverage_rows.append([P("Behavioral anomaly cross-check", compact_body_style if compact else body_style), P("Available", compact_body_style if compact else body_style), P(detail, compact_body_style if compact else body_style)])
        else:
            coverage_rows.append([P("Behavioral anomaly cross-check", compact_body_style if compact else body_style), P("Not applicable", compact_body_style if compact else body_style), P(sanitized_reason(generic.get("reason")), compact_body_style if compact else body_style)])

        if classifier.get("available"):
            coverage = classifier.get("feature_coverage")
            detail = (
                f"Applicable; {100 * float(coverage):.0f}% feature coverage"
                if coverage is not None
                else "Applicable"
            )
            coverage_rows.append([P("Fault-pattern cross-check", compact_body_style if compact else body_style), P("Available", compact_body_style if compact else body_style), P(detail, compact_body_style if compact else body_style)])
        else:
            coverage_rows.append([P("Fault-pattern cross-check", compact_body_style if compact else body_style), P("Not applicable", compact_body_style if compact else body_style), P(sanitized_reason(classifier.get("reason")), compact_body_style if compact else body_style)])

        widths = [108, 58, 134] if compact else [170, 90, 236]
        table = Table(coverage_rows, colWidths=widths)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF0FF")),
                    ("BOX", (0, 0), (-1, -1), 0.5, border),
                    ("INNERGRID", (0, 0), (-1, -1), 0.35, border),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4 if compact else 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4 if compact else 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 3.5 if compact else 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5 if compact else 5),
                ]
            )
        )
        return table

    def make_signature_table(compact=False):
        signature_rows = [
            [P("SIGNATURE", label_style), P("SCORE", label_style), P("STATE", label_style)]
            if compact
            else [P("SIGNATURE", label_style), P("EVIDENCE SCORE", label_style), P("APPLICABILITY", label_style)]
        ]

        for item in report.get("ranked_hypotheses", []):
            style = compact_body_style if compact else body_style
            signature_rows.append(
                [
                    P(item.get("name", "Unknown"), style),
                    P(f"{float(item.get('score', 0.0)):.2f}", style),
                    P("Checked" if item.get("available", True) else "N/A", style),
                ]
            )

        if len(signature_rows) == 1:
            signature_rows.append([P("No signature-ranking data"), P("-"), P("-")])

        widths = [188, 54, 62] if compact else [270, 100, 126]
        table = Table(signature_rows, colWidths=widths)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF0FF")),
                    ("BOX", (0, 0), (-1, -1), 0.5, border),
                    ("INNERGRID", (0, 0), (-1, -1), 0.35, border),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4 if compact else 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4 if compact else 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 3 if compact else 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3 if compact else 5),
                ]
            )
        )
        return table

    def make_values_table(compact=False):
        measured = top_hypothesis(report).get("evidence", {})
        evidence_rows = flatten_evidence(measured) if measured else []
        if not evidence_rows:
            return None

        values = [[P("SIGNAL", label_style), P("VALUE", label_style)]]
        style = compact_body_style if compact else body_style
        limit = 10 if compact else 28
        for row in evidence_rows[:limit]:
            values.append([P(row.get("Signal", ""), style), P(row.get("Value", ""), style)])

        widths = [126, 58] if compact else [320, 176]
        table = Table(values, colWidths=widths)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF0FF")),
                    ("BOX", (0, 0), (-1, -1), 0.5, border),
                    ("INNERGRID", (0, 0), (-1, -1), 0.35, border),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4 if compact else 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4 if compact else 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 3 if compact else 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3 if compact else 4),
                ]
            )
        )
        return table

    story = []

    # Brand mark: robot face inside a replay orbit.
    logo = Drawing(48, 48)
    logo.add(Rect(0, 0, 48, 48, rx=13, ry=13, fillColor=colors.HexColor("#14284A"), strokeColor=None))
    logo.add(Circle(24, 24, 16, fillColor=None, strokeColor=cyan, strokeWidth=2.4))
    logo.add(Rect(15, 17, 18, 15, rx=5, ry=5, fillColor=colors.HexColor("#1D3F7A"), strokeColor=colors.white, strokeWidth=1.6))
    logo.add(Circle(20, 25, 1.7, fillColor=cyan, strokeColor=None))
    logo.add(Circle(28, 25, 1.7, fillColor=cyan, strokeColor=None))
    logo.add(Line(20, 20, 28, 20, strokeColor=colors.white, strokeWidth=1.5))
    logo.add(Line(24, 32, 24, 36, strokeColor=colors.white, strokeWidth=1.4))
    logo.add(Circle(24, 38, 1.5, fillColor=cyan, strokeColor=None))
    # Replay arrow detail makes the mark distinctive even at small sizes.
    logo.add(Line(35, 38, 40, 38, strokeColor=cyan, strokeWidth=2.0))
    logo.add(Line(40, 38, 37.5, 40.5, strokeColor=cyan, strokeWidth=2.0))

    title_block = Table(
        [
            [
                logo,
                [
                    Paragraph("ROBOT<font color='#286BF1'>REPLAY</font>", title_style),
                    Paragraph("INCIDENT INVESTIGATION REPORT", subtitle_style),
                ],
            ]
        ],
        colWidths=[55, 440],
    )
    title_block.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(title_block)
    story.append(Spacer(1, 4))
    story.append(
        P(
            f"Mission {mission} | Evidence-backed reconstruction of what failed, when it failed, and what to inspect next.",
            subtitle_style,
        )
    )

    metric_data = [
        [
            P("MISSION STATUS", label_style),
            P("ROOT CAUSE", label_style),
            P("FIRST EVIDENCE", label_style),
            P("EVIDENCE STRENGTH", label_style),
        ],
        [
            P(status, status_metric_style),
            P(cause, metric_style),
            P(f"{float(event):.3f} s" if event is not None else "-", metric_style),
            P(f"{evidence_label(evidence_score)} ({evidence_score:.2f})", metric_style),
        ],
    ]
    metric_table = Table(metric_data, colWidths=[96, 190, 100, 110], rowHeights=[18, 36])
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), pale),
                ("BACKGROUND", (0, 1), (-1, 1), status_fill),
                ("BOX", (0, 0), (-1, -1), 0.7, border),
                ("LINEBEFORE", (0, 0), (0, -1), 3.0, status_accent),
                ("INNERGRID", (0, 0), (-1, -1), 0.45, border),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(metric_table)

    metadata = []
    storage = bag.get("storage_id", bag.get("storage", "-"))
    metadata.append(f"Storage: {storage}")
    if bag.get("duration_s") is not None:
        metadata.append(f"Duration: {float(bag.get('duration_s')):.1f} s")
    if bag.get("message_count") is not None:
        metadata.append(f"Messages: {int(bag.get('message_count')):,}")
    if perf.get("processing_time_s") is not None:
        metadata.append(f"Analysis time: {float(perf.get('processing_time_s')):.2f} s")
    story.append(Spacer(1, 4))
    story.append(P(" | ".join(metadata), small_style))

    story.append(P("Executive Summary", heading_style))
    story.append(P(incident_summary(report)))

    supporting = report.get("supporting_evidence", [])
    chain = report.get("failure_chain", [])
    checks = report.get("recommended_checks", [])

    if is_healthy:
        # Compact healthy layout: keep the report to one page whenever the data volume allows it.
        assessment_rows = [
            [P("FAILURE SEQUENCE", label_style), P("No supported failure propagation was reconstructed.", compact_body_style)],
            [P("ENGINEERING ACTION", label_style), P("No corrective action is indicated by the supported signatures.", compact_body_style)],
        ]
        assessment = Table(assessment_rows, colWidths=[104, 392])
        assessment.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), green_soft),
                    ("BOX", (0, 0), (-1, -1), 0.45, border),
                    ("INNERGRID", (0, 0), (-1, -1), 0.30, border),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(assessment)

        story.append(P("Supporting Evidence", compact_heading_style))
        if supporting:
            for item in supporting[:3]:
                story.append(bullet(item, compact_body_style))
        else:
            story.append(P("No supported failure signature was identified in the available diagnostic coverage.", compact_body_style))

        story.append(P("Telemetry Evidence Snapshot", compact_heading_style))
        story.append(build_pdf_telemetry_drawing(report, features, compact=True))

        story.append(P("Technical Appendix", compact_heading_style))
        left_cell = [P("Diagnostic Coverage", compact_heading_style), make_coverage_table(compact=True), Spacer(1, 3), P("Fault Signatures Checked", compact_heading_style), make_signature_table(compact=True)]

        right_cell = []
        values_table = make_values_table(compact=True)
        if values_table is not None:
            right_cell.extend([P("Measured Evidence", compact_heading_style), values_table, Spacer(1, 4)])
        right_cell.extend(
            [
                P("Scope and Interpretation", compact_heading_style),
                P(
                    "A healthy result means no supported signature was established in the available diagnostic coverage; it is not a guarantee that every possible robot failure is absent. Evidence strength is a heuristic diagnostic score, not a calibrated probability.",
                    compact_body_style,
                ),
            ]
        )

        appendix_grid = Table([[left_cell, right_cell]], colWidths=[307, 189])
        appendix_grid.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (0, 0), 8),
                    ("RIGHTPADDING", (1, 0), (1, 0), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(appendix_grid)

    else:
        story.append(P("Failure Timeline", heading_style))
        if event is not None:
            story.append(bullet(f"{float(event):.3f} s - First supported diagnostic evidence appears."))
        if chain:
            for index, step in enumerate(chain, start=1):
                story.append(bullet(f"Step {index}: {step}"))
        else:
            story.append(P("No failure propagation sequence was reconstructed."))

        story.append(P("Supporting Evidence", heading_style))
        if supporting:
            for item in supporting:
                story.append(bullet(item))
        else:
            story.append(P("No supporting evidence summary was available."))

        story.append(P("Telemetry Evidence Snapshot", heading_style))
        story.append(build_pdf_telemetry_drawing(report, features, compact=False))
        if event is not None:
            story.append(Spacer(1, 3))
            story.append(P("The shaded marker region highlights the first supported diagnostic event for visual reference.", small_style))

        # Keep the full action block together. For normal fault reports it starts page 2.
        if checks:
            story.append(PageBreak())

        action_flowables = [P("Recommended Engineering Actions", heading_style)]
        if checks:
            action_rows = []
            for index, check in enumerate(checks, start=1):
                action_rows.append([P(str(index), metric_style), P(check)])
            actions = Table(action_rows, colWidths=[28, 468])
            actions.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BACKGROUND", (0, 0), (0, -1), coral_soft if is_anomalous else colors.HexColor("#EAF0FF")),
                        ("BOX", (0, 0), (-1, -1), 0.5, border),
                        ("INNERGRID", (0, 0), (-1, -1), 0.35, border),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            action_flowables.append(actions)
        else:
            action_flowables.append(P("No recommended engineering actions were generated."))
        story.append(KeepTogether(action_flowables))

        story.append(Spacer(1, 7))
        story.append(P("Technical Appendix", title_style))
        story.append(P("Diagnostic Coverage", heading_style))
        story.append(make_coverage_table(compact=False))
        story.append(P("Fault Signatures Checked", heading_style))
        story.append(make_signature_table(compact=False))

        values_table = make_values_table(compact=False)
        if values_table is not None:
            story.append(P("Measured Evidence Values", heading_style))
            story.append(values_table)

        story.append(P("Scope and Interpretation", heading_style))
        story.append(
            P(
                "RobotReplay reports evidence for the diagnostic signatures applicable to the telemetry available in this recording. "
                "An unresolved result is not forced into a known signature. Evidence strength is a heuristic diagnostic score and is not a calibrated probability."
            )
        )

    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return buffer.getvalue()

def render_pdf_download(report: dict, features: pd.DataFrame) -> None:
    render_html(
        """
        <div class="rr-report-cta">
            <div class="rr-report-badge">Primary export</div>
            <div class="rr-report-title">Incident Investigation Report</div>
            <div class="rr-report-copy">
                Generate a professional PDF with the executive summary, root cause,
                failure timeline, telemetry evidence, recommended actions, diagnostic
                coverage, and technical appendix.
            </div>
            <div class="rr-report-meta">Designed for engineering handoff, incident review, and technical analysis.</div>
        </div>
        """
    )

    try:
        pdf_bytes = build_incident_pdf(report, features)
        st.download_button(
            "Download PDF Incident Report",
            data=pdf_bytes,
            file_name=f"{report.get('mission', 'mission')}_incident_report.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"pdf_{report.get('mission', 'mission')}_{id(report)}",
        )
    except Exception as exc:
        st.error(f"PDF report could not be generated: {exc}")


def render_advanced(report: dict, features: pd.DataFrame) -> None:
    with st.expander("Advanced engineering details", expanded=False):
        st.markdown("### Diagnostic coverage")
        st.caption(
            "These details describe which analysis paths were applicable to the telemetry in this recording."
        )

        generic = report.get("generic_ml", {})
        classifier = report.get("fault_classifier", {})

        col1, col2 = st.columns(2)

        with col1:
            if generic.get("available"):
                coverage = generic.get("feature_coverage")
                coverage_text = f"{100 * float(coverage):.0f}% feature coverage" if coverage is not None else "Applicable"
                st.success(f"Behavioral anomaly cross-check: available · {coverage_text}")
            else:
                st.info(
                    "Behavioral anomaly cross-check: not applicable · "
                    + sanitized_reason(generic.get("reason"))
                )

        with col2:
            if classifier.get("available"):
                coverage = classifier.get("feature_coverage")
                coverage_text = f"{100 * float(coverage):.0f}% feature coverage" if coverage is not None else "Applicable"
                st.success(f"Fault-pattern cross-check: available · {coverage_text}")
            else:
                st.info(
                    "Fault-pattern cross-check: not applicable · "
                    + sanitized_reason(classifier.get("reason"))
                )

        agreement = report.get("intelligence_agreement", {})
        if agreement.get("available"):
            st.write(f"Cross-check state: **{agreement.get('state', 'UNKNOWN').replace('_', ' ').title()}**")

        st.markdown("### Evidence values")
        measured = top_hypothesis(report).get("evidence", {})
        rows = flatten_evidence(measured) if measured else []

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No measured evidence table is available for this result.")

        st.markdown("### Other signatures checked")
        st.caption("Scores below are heuristic evidence scores, not probabilities.")
        render_other_signatures(report)

        st.markdown("### Engineering exports")
        download_json, download_csv = st.columns(2)

        with download_json:
            st.download_button(
                "Download incident JSON",
                data=json.dumps(report, indent=2),
                file_name=f"{report.get('mission', 'mission')}_incident.json",
                mime="application/json",
                use_container_width=True,
            )

        with download_csv:
            st.download_button(
                "Download telemetry CSV",
                data=features.to_csv(index=False),
                file_name=f"{report.get('mission', 'mission')}_telemetry.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("### Raw incident data")
        st.json(report, expanded=False)


def render_report(report: dict, features: pd.DataFrame) -> None:
    status = report.get("mission_status", "UNKNOWN")
    cause = report.get("root_cause", "Unknown")
    evidence_score = float(report.get("evidence_strength", 0.0))
    event = report.get("first_abnormal_time_s")
    perf = report.get("performance", {})
    bag = report.get("bag", {})

    if status == "HEALTHY":
        status_class = "rr-status-healthy"
    elif status.startswith("ANOMALOUS"):
        status_class = "rr-status-anomalous"
    else:
        status_class = "rr-status-unknown"

    processing = perf.get("processing_time_s")
    realtime = perf.get("realtime_factor")
    storage = bag.get("storage_id", bag.get("storage", "—"))

    meta_parts = [
        f"Mission: {report.get('mission', '—')}",
        f"Storage: {storage}",
    ]

    if processing is not None:
        meta_parts.append(f"Analysis: {float(processing):.2f} s")
    if realtime is not None:
        meta_parts.append(f"{realtime}× realtime")

    result_class = (
        "result-healthy"
        if status == "HEALTHY"
        else "result-anomalous"
        if status.startswith("ANOMALOUS")
        else "result-unknown"
    )

    render_html(
        f"""
        <div class="rr-result-shell {result_class}">
            <div class="rr-result-top">
                <div>
                    <div class="rr-result-label">Incident summary</div>
                    <div class="rr-result-cause">{safe_text(cause)}</div>
                    <div class="rr-result-meta">{safe_text(' · '.join(meta_parts))}</div>
                </div>
                <div class="rr-status-badge {status_class}">{safe_text(status)}</div>
            </div>
        </div>
        """
    )

    render_crosscheck(report)

    metric_cols = st.columns(4)

    with metric_cols[0]:
        render_metric_card(
            "Status",
            status,
            "Mission assessment",
        )

    with metric_cols[1]:
        render_metric_card(
            "First evidence",
            f"{float(event):.3f} s" if event is not None else "—",
            "First supported diagnostic event",
        )

    with metric_cols[2]:
        render_metric_card(
            "Evidence strength",
            evidence_label(evidence_score),
            f"Heuristic evidence score {evidence_score:.2f}",
        )

    with metric_cols[3]:
        render_metric_card(
            "Recording",
            f"{float(bag.get('duration_s', 0)):.1f} s",
            f"{int(bag.get('message_count', 0)):,} messages",
        )

    st.write("")
    render_what_happened(report)
    st.write("")
    render_pdf_download(report, features)
    st.write("")

    evidence_col, telemetry_col = st.columns([0.9, 1.35])

    with evidence_col:
        st.markdown("### Evidence")
        st.caption("Evidence supporting the incident diagnosis.")
        render_supporting_evidence(report)

    with telemetry_col:
        st.markdown("### Telemetry")
        st.caption("Signals most relevant to the diagnosed incident.")
        st.plotly_chart(
            build_telemetry_chart(report, features),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

    st.divider()

    timeline_col, action_col = st.columns([1, 1])

    with timeline_col:
        st.markdown("### Failure timeline")
        st.caption("Reconstructed progression from incident onset through downstream impact.")
        render_failure_timeline(report)

    with action_col:
        st.markdown("### Recommended actions")
        st.caption("Engineering checks prioritized from the available evidence.")
        render_actions(report)

    st.divider()
    render_advanced(report, features)


def load_sample(mission: str):
    report_path = RESULTS_DIR / f"{mission}_incident.json"
    feature_path = RUNTIME_DIR / f"{mission}_features.csv"

    report = json.loads(report_path.read_text())
    features = pd.read_csv(feature_path)

    return report, features


def clear_upload_result() -> None:
    for key in [
        "rr_uploaded_report",
        "rr_uploaded_features",
        "rr_uploaded_filename",
    ]:
        if key in st.session_state:
            del st.session_state[key]


render_header()

analyze_tab, sample_tab = st.tabs([
    "Analyze Recording",
    "Sample Incidents",
])


with analyze_tab:
    render_html(
        """
        <div class="rr-upload-hero">
            <div class="rr-upload-copy-card">
                <div class="rr-eyebrow">ROS 2 incident investigation</div>
                <div class="rr-upload-heading">
                    Turn a robot recording into an evidence-backed incident report.
                </div>
                <div class="rr-upload-desc">
                    Upload one ROS 2 bag. RobotReplay reconstructs the mission,
                    identifies supported failure signatures, traces the failure chain,
                    and surfaces the evidence an engineer needs to investigate the incident.
                </div>
            </div>
            <div class="rr-flow-card">
                <div class="rr-flow-title">Investigation workflow</div>
                <div class="rr-flow-item"><div class="rr-flow-number">1</div>Validate recording</div>
                <div class="rr-flow-item"><div class="rr-flow-number">2</div>Reconstruct synchronized telemetry</div>
                <div class="rr-flow-item"><div class="rr-flow-number">3</div>Identify abnormal behavior</div>
                <div class="rr-flow-item"><div class="rr-flow-number">4</div>Correlate fault evidence</div>
                <div class="rr-flow-item"><div class="rr-flow-number">5</div>Reconstruct failure and actions</div>
            </div>
        </div>
        """
    )

    render_html(
        """
        <div class="rr-trust-row">
            <div class="rr-trust-item"><span class="rr-trust-icon">✓</span>Local processing</div>
            <div class="rr-trust-item"><span class="rr-trust-icon">✓</span>Secure ZIP validation</div>
            <div class="rr-trust-item"><span class="rr-trust-icon">✓</span>SQLite3 + MCAP</div>
            <div class="rr-trust-item"><span class="rr-trust-icon">✓</span>Evidence-based diagnosis</div>
        </div>
        """
    )

    uploaded = st.file_uploader(
        "Upload ROS 2 bag",
        type=["zip"],
        accept_multiple_files=False,
        help=(
            "ZIP one ROS 2 bag directory containing metadata.yaml and its "
            ".db3 or .mcap recording. Maximum compressed upload size: 250 MiB."
        ),
    )

    st.caption("Processed locally and not retained after analysis.")

    if uploaded is not None:
        st.caption(f"{uploaded.name} · {format_bytes(uploaded.size)}")

        previous = st.session_state.get("rr_uploaded_filename")

        if previous is not None and previous != uploaded.name:
            clear_upload_result()

        if st.button(
            "Analyze Recording",
            type="primary",
            use_container_width=True,
        ):
            clear_upload_result()

            progress = st.progress(0, text="Starting incident investigation...")

            try:
                payload = uploaded.getvalue()

                progress.progress(10, text="Validating recording...")

                with tempfile.TemporaryDirectory(prefix="robotreplay_upload_") as temp_dir:
                    temp_root = Path(temp_dir)

                    upload_result = safe_extract_rosbag_zip(
                        payload,
                        temp_root / "bag",
                    )

                    progress.progress(28, text="Reconstructing synchronized telemetry...")

                    started = time.perf_counter()

                    report, features = analyze_bag(
                        upload_result.bag_dir,
                        runtime_dir=temp_root / "runtime",
                        results_dir=None,
                        persist=False,
                    )

                    elapsed = time.perf_counter() - started

                    progress.progress(72, text="Correlating incident evidence...")
                    progress.progress(88, text="Reconstructing failure sequence...")

                    report["upload"] = {
                        "original_filename": uploaded.name,
                        "upload_size_bytes": upload_result.upload_size_bytes,
                        "temporary_processing": True,
                        "retained_after_analysis": False,
                        "ui_processing_time_s": round(elapsed, 3),
                    }

                    st.session_state["rr_uploaded_report"] = report
                    st.session_state["rr_uploaded_features"] = features
                    st.session_state["rr_uploaded_filename"] = uploaded.name

                progress.progress(100, text="Incident investigation complete.")
                time.sleep(0.15)
                progress.empty()
                st.success("Analysis complete.")

            except (UploadSecurityError, BagValidationError) as exc:
                progress.empty()
                st.error(f"Upload rejected: {exc}")

            except AnalysisError as exc:
                progress.empty()
                st.error(f"Analysis failed: {exc}")

            except Exception as exc:
                progress.empty()
                st.exception(exc)

    if (
        "rr_uploaded_report" in st.session_state
        and "rr_uploaded_features" in st.session_state
    ):
        st.divider()
        st.markdown("## Incident Investigation")
        render_report(
            st.session_state["rr_uploaded_report"],
            st.session_state["rr_uploaded_features"],
        )


with sample_tab:
    st.markdown("## Sample Incidents")
    st.caption(
        "Explore pre-analyzed recordings that demonstrate the current diagnostic signatures."
    )

    selected = st.selectbox(
        "Incident",
        list(SAMPLE_MISSIONS.keys()),
    )

    selected_info = SAMPLE_MISSIONS[selected]
    st.caption(selected_info["note"])

    mission = selected_info["mission"]

    try:
        report, features = load_sample(mission)
        render_report(report, features)

        if mission.startswith("Test_3-"):
            st.caption(
                "This public external recording was used during development of the "
                "LiDAR Stream Integrity Failure signature; it is not independent validation of that signature."
            )

    except Exception as exc:
        st.error(f"Sample incident unavailable: {exc}")


render_html(
    """
    <div class="rr-footer">
        <div>RobotReplay · Autonomous Robot Incident Intelligence</div>
        <div>Black-box replay · Explainable evidence · PDF investigation reports</div>
    </div>
    """
)
