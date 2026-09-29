"""
app.py — Streamlit frontend for the Human Activity Recognition (HAR) system.

Run:
    streamlit run frontend/app.py

Environment variables:
    API_URL  — FastAPI backend URL (default: http://localhost:8000)
"""

import os
import time
from io import StringIO
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")

ACTIVITY_ICONS = {
    "WALKING": "🚶",
    "WALKING_UPSTAIRS": "🪜",
    "WALKING_DOWNSTAIRS": "⬇️",
    "SITTING": "🪑",
    "STANDING": "🧍",
    "LAYING": "🛌",
}

ACTIVITY_COLORS = {
    "WALKING": "#00d4ff",
    "WALKING_UPSTAIRS": "#7b61ff",
    "WALKING_DOWNSTAIRS": "#ff6b6b",
    "SITTING": "#ffd93d",
    "STANDING": "#6bcb77",
    "LAYING": "#ff922b",
}

SIGNAL_NAMES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

SIGNAL_COLORS = [
    "#00d4ff",
    "#7b61ff",
    "#ff6b6b",
    "#ffd93d",
    "#6bcb77",
    "#ff922b",
    "#a8dadc",
    "#e9c46a",
    "#f4a261",
]

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HAR System — Activity Recognition",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Styling ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* ── Global background ── */
    .stApp { background: #0d1117; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b27 0%, #0d1117 100%);
        border-right: 1px solid #21262d;
    }

    /* ── Remove default padding ── */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* ── Custom metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #161b27 0%, #1c2333 100%);
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7b61ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .label {
        color: #8b949e;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.25rem;
    }

    /* ── Result card ── */
    .result-card {
        background: linear-gradient(135deg, #161b27 0%, #1c2333 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        animation: fadeInUp 0.5s ease;
    }
    .result-card .activity-name {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
    }
    .result-card .activity-icon {
        font-size: 4rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    .result-card .confidence-bar {
        height: 8px;
        border-radius: 4px;
        margin-top: 1rem;
        background: linear-gradient(90deg, #00d4ff, #7b61ff);
    }

    /* ── Upload zone ── */
    .uploadedFile { display: none; }
    [data-testid="stFileUploadDropzone"] {
        background: #161b27 !important;
        border: 2px dashed #30363d !important;
        border-radius: 12px !important;
        transition: border-color 0.2s ease !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #00d4ff !important;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e6edf3;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #21262d;
    }

    /* ── Status pill ── */
    .pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .pill-green  { background: rgba(107, 203, 119, 0.2); color: #6bcb77; border: 1px solid #6bcb77; }
    .pill-yellow { background: rgba(255, 217, 61, 0.2);  color: #ffd93d; border: 1px solid #ffd93d; }
    .pill-red    { background: rgba(255, 107, 107, 0.2); color: #ff6b6b; border: 1px solid #ff6b6b; }

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0);    }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.6; }
    }

    /* ── Streamlit default overrides ── */
    h1, h2, h3, h4, h5 { color: #e6edf3 !important; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #00d4ff; }
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    div[data-testid="metric-container"] {
        background: #161b27;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper Functions ──────────────────────────────────────────────────────────


def api_get(endpoint: str, params: dict = None) -> Optional[Dict]:
    """Make a GET request to the API and return JSON or None on error."""
    try:
        resp = requests.get(f"{API_URL}{endpoint}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def check_health() -> Dict:
    """Return health status dict (uses cached API call)."""
    data = api_get("/health")
    if data is None:
        return {"status": "unreachable", "model_loaded": False, "database": "error"}
    return data


def render_plotly_dark(fig: go.Figure) -> go.Figure:
    """Apply consistent dark theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b27",
        font=dict(color="#e6edf3", family="Inter"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#21262d",
        ),
    )
    fig.update_xaxes(gridcolor="#21262d", zerolinecolor="#21262d")
    fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#21262d")
    return fig


def render_probability_chart(probabilities: Dict[str, float]) -> go.Figure:
    """Render a horizontal bar chart of per-class probabilities."""
    labels = list(probabilities.keys())
    values = [probabilities[k] * 100 for k in labels]
    colors = [ACTIVITY_COLORS.get(k, "#00d4ff") for k in labels]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=0),
            ),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(color="#e6edf3", size=12),
        )
    )
    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Probability (%)",
        xaxis=dict(range=[0, 115]),
        height=320,
        showlegend=False,
    )
    return render_plotly_dark(fig)


def render_sensor_chart(df: pd.DataFrame) -> go.Figure:
    """Render an interactive multi-line Plotly chart of all 9 sensor signals."""
    fig = go.Figure()
    for i, col in enumerate(df.columns[:9]):
        name = SIGNAL_NAMES[i] if i < len(SIGNAL_NAMES) else col
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=df.iloc[:, i],
                name=name,
                line=dict(color=SIGNAL_COLORS[i % len(SIGNAL_COLORS)], width=1.5),
                opacity=0.85,
            )
        )
    fig.update_layout(
        title="Sensor Signal Preview (128 time steps × 9 channels)",
        xaxis_title="Time Step",
        yaxis_title="Signal Value",
        height=400,
        hovermode="x unified",
    )
    return render_plotly_dark(fig)


def render_distribution_pie(distribution: List[Dict]) -> go.Figure:
    """Render a donut pie chart of activity distribution."""
    labels = [d["activity"] for d in distribution]
    values = [d["count"] for d in distribution]
    colors = [ACTIVITY_COLORS.get(lbl, "#00d4ff") for lbl in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="#0d1117", width=2)),
            textfont=dict(color="#e6edf3"),
        )
    )
    fig.update_layout(title="Activity Distribution", height=360)
    return render_plotly_dark(fig)


def render_confidence_timeline(history: List[Dict]) -> go.Figure:
    """Render confidence score over time for recent predictions."""
    if not history:
        return go.Figure()
    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    fig = go.Figure(
        go.Scatter(
            x=df["timestamp"],
            y=df["confidence"] * 100,
            mode="lines+markers",
            line=dict(color="#00d4ff", width=2),
            marker=dict(size=8, color="#7b61ff"),
            hovertemplate="<b>%{text}</b><br>Confidence: %{y:.1f}%<extra></extra>",
            text=df.get("predicted_activity", [""] * len(df)),
        )
    )
    fig.update_layout(
        title="Confidence Score Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 105]),
        height=300,
    )
    return render_plotly_dark(fig)


# ── Sidebar ───────────────────────────────────────────────────────────────────


def render_sidebar(health: Dict) -> str:
    """Render the sidebar and return the active page name."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 1rem 0 0.5rem;">
                <span style="font-size:3rem;">🏃</span>
                <h2 style="color:#e6edf3; margin:0.25rem 0 0;">HAR System</h2>
                <p style="color:#8b949e; font-size:0.8rem; margin:0;">
                    CNN + LSTM · UCI HAR Dataset
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # Navigation
        page = st.radio(
            "Navigation",
            ["🏠 Home & Predict", "📊 Analytics", "ℹ️ Model Info"],
            label_visibility="collapsed",
        )

        st.divider()

        # API Status
        st.markdown("**API Status**")
        status = health.get("status", "unreachable")
        model_ok = health.get("model_loaded", False)
        db_ok = health.get("database", "error") == "connected"

        if status == "ok":
            st.markdown(
                '<span class="pill pill-green">● API Online</span>',
                unsafe_allow_html=True,
            )
        elif status == "degraded":
            st.markdown(
                '<span class="pill pill-yellow">◐ Degraded</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="pill pill-red">● Offline</span>', unsafe_allow_html=True
            )

        col1, col2 = st.columns(2)
        with col1:
            icon = "✅" if model_ok else "❌"
            st.caption(f"{icon} Model")
        with col2:
            icon = "✅" if db_ok else "❌"
            st.caption(f"{icon} Database")

        if status == "unreachable":
            st.warning(
                "Backend not reachable.\n\n"
                "Start it with:\n```\nuvicorn backend.main:app --reload\n```",
                icon="⚠️",
            )
        elif not model_ok:
            st.info(
                "Model not trained yet.\n\n" "Run:\n```\npython model/train.py\n```",
                icon="ℹ️",
            )

        st.divider()
        st.caption(f"API: `{API_URL}`")
        st.caption("v1.0.0 · HAR CNN+LSTM")

    return page


# ── Pages ─────────────────────────────────────────────────────────────────────


def page_home(health: Dict) -> None:
    """Home page: upload CSV, predict, display results."""

    # ── Page Header ────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
            <h1 style="font-size:2rem; font-weight:800; margin-bottom:0.25rem;">
                🏃 Human Activity Recognition
            </h1>
            <p style="color:#8b949e; font-size:1rem; margin:0;">
                Upload a sensor data CSV to predict human activity using CNN + LSTM deep learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Quick Stats ─────────────────────────────────────────────────────
    history_data = api_get("/history", {"limit": 1})
    total_preds = history_data["total"] if history_data else 0

    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("🎯", str(total_preds), "Total Predictions"),
        ("🧠", "CNN+LSTM", "Model Architecture"),
        ("📡", "9 Channels", "Sensor Inputs"),
        ("🏆", "6 Activities", "Target Classes"),
    ]
    for col, (icon, value, label) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div class="value">{value}</div>
                    <div class="label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Upload & Predict ─────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(
            '<div class="section-header">📂 Upload Sensor Data</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background:#161b27; border:1px solid #21262d; border-radius:10px;
                        padding:1rem; margin-bottom:1rem; font-size:0.85rem; color:#8b949e;">
                <b style="color:#e6edf3;">CSV Format Requirements</b><br>
                • <b style="color:#00d4ff;">Any number of rows</b> — auto-padded or windowed<br>
                • Exactly <b style="color:#00d4ff;">9 columns</b> (sensor channels)<br>
                • Column order: body_acc_x/y/z · body_gyro_x/y/z · total_acc_x/y/z<br>
                • No header required (numeric data only is fine)
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Drop your CSV here",
            type=["csv"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            try:
                df_preview = pd.read_csv(uploaded_file, header=None)
                uploaded_file.seek(0)
                rows, cols = df_preview.shape
                # Only enforce column count — row count is handled by the backend
                col_ok = cols == 9

                if col_ok:
                    if rows == 128:
                        st.success(f"✅ Ready: {rows} rows × {cols} columns", icon="✅")
                    elif rows < 128:
                        st.info(
                            f"ℹ️ {rows} rows × {cols} columns detected. "
                            f"The backend will zero-pad to 128 rows automatically.",
                            icon="ℹ️",
                        )
                    else:
                        n_windows = rows // 128 + (1 if rows % 128 else 0)
                        st.info(
                            f"ℹ️ {rows} rows × {cols} columns detected. "
                            f"The backend will split into {n_windows} window(s) and "
                            f"apply majority voting automatically.",
                            icon="ℹ️",
                        )
                    shape_ok = True
                else:
                    st.error(
                        f"❌ Column mismatch: {cols} column(s) found, need exactly 9. "
                        f"Expected: body_acc_x/y/z · body_gyro_x/y/z · total_acc_x/y/z",
                        icon="❌",
                    )
                    shape_ok = False

            except Exception as e:
                st.error(f"Cannot parse file: {e}")
                df_preview = None
                shape_ok = False
        else:
            df_preview = None
            shape_ok = False

        # Predict Button
        model_ready = health.get("model_loaded", False)
        can_predict = uploaded_file is not None and shape_ok and model_ready

        if st.button(
            "🔮 Predict Activity",
            disabled=not can_predict,
            use_container_width=True,
            type="primary",
        ):
            with st.spinner("🧠 Running inference …"):
                uploaded_file.seek(0)
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")
                }
                try:
                    resp = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                    resp.raise_for_status()
                    prediction = resp.json()
                    st.session_state["last_prediction"] = prediction
                    st.session_state["last_df"] = df_preview
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Is it running?")
                    prediction = None
                except Exception as exc:
                    try:
                        detail = resp.json().get("detail", str(exc))
                    except Exception:
                        detail = str(exc)
                    st.error(f"Prediction error: {detail}")
                    prediction = None

        if not model_ready and uploaded_file:
            st.warning(
                "Model not trained yet. Run `python model/train.py` first.",
                icon="⚠️",
            )

    # ── Results ──────────────────────────────────────────────────────────
    with right:
        st.markdown(
            '<div class="section-header">🎯 Prediction Result</div>',
            unsafe_allow_html=True,
        )

        pred = st.session_state.get("last_prediction")

        if pred:
            activity = pred["predicted_activity"]
            confidence = pred["confidence"]  # 0-1 float
            conf_pct = confidence * 100
            icon = ACTIVITY_ICONS.get(activity, "🤸")
            color = ACTIVITY_COLORS.get(activity, "#00d4ff")
            n_windows = pred.get("windows_count", 1)
            note = pred.get("preprocessing_note", "")
            proc_ms = int(pred.get("processing_time_ms") or 0)
            pred_id = pred.get("id", "?")

            # ── Activity icon (single-line, safe) ─────────────────────────
            st.markdown(
                f'<div style="text-align:center;padding:1.25rem 0 0.25rem;">'
                f'<span style="font-size:4.5rem;line-height:1;">{icon}</span></div>',
                unsafe_allow_html=True,
            )

            # ── Activity name (single-line, safe) ─────────────────────────
            st.markdown(
                f'<div style="text-align:center;font-size:1.9rem;font-weight:800;'
                f'color:{color};margin-bottom:0.75rem;letter-spacing:0.01em;">'
                f"{activity}</div>",
                unsafe_allow_html=True,
            )

            # ── Confidence progress bar (native) ──────────────────────────
            st.progress(confidence, text=f"Confidence: **{conf_pct:.1f}%**")

            # ── Key metrics row (native) ──────────────────────────────────
            m1, m2, m3 = st.columns(3)
            m1.metric("🎯 Confidence", f"{conf_pct:.1f}%")
            m2.metric("⚡ Inference", f"{proc_ms} ms")
            m3.metric("🪟 Windows", str(n_windows))

            # ── Prediction ID caption ────────────────────────────────────
            st.caption(f"Prediction ID #{pred_id}")

            # ── Preprocessing note (native info box) ──────────────────────
            if note and "no adjustment" not in note.lower():
                st.info(f"🔧 **Auto-preprocessing applied** — {note}", icon="ℹ️")

            # ── Per-window breakdown (only when > 1 window) ───────────────
            window_preds = pred.get("window_predictions", [])
            if len(window_preds) > 1:
                with st.expander(
                    f"📋 Per-window breakdown ({n_windows} windows)", expanded=False
                ):
                    win_rows = [
                        {
                            "Window": f"#{wp['window_index']}",
                            "Activity": f"{ACTIVITY_ICONS.get(wp['predicted_activity'], '')} {wp['predicted_activity']}",
                            "Confidence": f"{wp['confidence'] * 100:.1f}%",
                        }
                        for wp in window_preds
                    ]
                    st.dataframe(
                        pd.DataFrame(win_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

            # ── Probabilities chart ───────────────────────────────────────
            st.plotly_chart(
                render_probability_chart(pred["probabilities"]),
                use_container_width=True,
            )

        else:
            st.markdown(
                '<div style="text-align:center;padding:3rem;color:#6e7681;'
                'border:1px dashed #21262d;border-radius:12px;">'
                '<div style="font-size:3rem;margin-bottom:0.75rem;">🎯</div>'
                '<div style="font-size:1rem;">Upload a CSV and click<br><b>Predict Activity</b></div>'
                "</div>",
                unsafe_allow_html=True,
            )

    # ── Sensor Preview ────────────────────────────────────────────────────
    df_display = st.session_state.get("last_df")
    if df_display is not None:
        st.divider()
        st.markdown(
            '<div class="section-header">📡 Sensor Data Preview</div>',
            unsafe_allow_html=True,
        )
        tab1, tab2 = st.tabs(["📈 Interactive Chart", "🔢 Raw Data"])
        with tab1:
            st.plotly_chart(render_sensor_chart(df_display), use_container_width=True)
        with tab2:
            display_df = df_display.copy()
            display_df.columns = SIGNAL_NAMES
            st.dataframe(
                display_df.head(20).round(4),
                use_container_width=True,
                height=350,
            )
            st.caption(
                f"Showing first 20 of {len(display_df)} row(s) · {display_df.shape[1]} channels"
            )

    # ── Prediction History ─────────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div class="section-header">🕐 Prediction History</div>',
        unsafe_allow_html=True,
    )

    hist = api_get("/history", {"limit": 10})
    if hist and hist["predictions"]:
        rows = []
        for p in hist["predictions"]:
            act = p["predicted_activity"]
            rows.append(
                {
                    "ID": p["id"],
                    "File": p["filename"],
                    "Activity": f"{ACTIVITY_ICONS.get(act, '')} {act}",
                    "Confidence": f"{p['confidence'] * 100:.1f}%",
                    "Time (ms)": p.get("processing_time_ms", "-"),
                    "Timestamp": p["timestamp"][:19].replace("T", " "),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"Showing last 10 of {hist['total']} predictions")
    elif hist is not None:
        st.info("No predictions yet. Upload a CSV to get started.", icon="ℹ️")
    else:
        st.warning("Cannot reach backend.", icon="⚠️")


def page_analytics() -> None:
    """Analytics dashboard page."""
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
            <h1 style="font-size:2rem; font-weight:800; margin-bottom:0.25rem;">
                📊 Analytics Dashboard
            </h1>
            <p style="color:#8b949e;">Aggregated statistics across all predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    analytics = api_get("/analytics", {"latest_n": 10})
    history = api_get("/history", {"limit": 100})

    if analytics is None:
        st.error("Cannot connect to backend.", icon="🔴")
        return

    total = analytics["total_predictions"]
    avg_conf = analytics["avg_confidence"] * 100
    dist = analytics["activity_distribution"]
    latest = analytics["latest_predictions"]

    # ── KPI Row ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    kpis = [
        ("🎯", str(total), "Total Predictions"),
        ("📈", f"{avg_conf:.1f}%", "Avg. Confidence"),
        ("🏷️", str(len(dist)), "Unique Activities"),
    ]
    for col, (icon, value, label) in zip([col1, col2, col3], kpis):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div class="value">{value}</div>
                    <div class="label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Charts Row ───────────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        if dist:
            st.plotly_chart(render_distribution_pie(dist), use_container_width=True)
        else:
            st.info("No prediction data yet.")

    with right:
        if history and history["predictions"]:
            st.plotly_chart(
                render_confidence_timeline(history["predictions"]),
                use_container_width=True,
            )
        else:
            st.info("No timeline data yet.")

    # ── Activity Bar Chart ────────────────────────────────────────────────
    if dist:
        st.divider()
        labels = [d["activity"] for d in dist]
        counts = [d["count"] for d in dist]
        colors = [ACTIVITY_COLORS.get(lbl, "#00d4ff") for lbl in labels]

        fig = go.Figure(
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color=colors, line=dict(width=0)),
                text=counts,
                textposition="outside",
                textfont=dict(color="#e6edf3"),
            )
        )
        fig.update_layout(
            title="Prediction Count per Activity",
            xaxis_title="Activity",
            yaxis_title="Count",
            height=350,
            showlegend=False,
        )
        st.plotly_chart(render_plotly_dark(fig), use_container_width=True)

    # ── Recent Predictions Table ───────────────────────────────────────────
    if latest:
        st.divider()
        st.markdown(
            '<div class="section-header">🕐 Latest Predictions</div>',
            unsafe_allow_html=True,
        )
        rows = []
        for p in latest:
            act = p["predicted_activity"]
            rows.append(
                {
                    "ID": p["id"],
                    "File": p["filename"],
                    "Activity": f"{ACTIVITY_ICONS.get(act, '')} {act}",
                    "Confidence": f"{p['confidence'] * 100:.1f}%",
                    "Timestamp": p["timestamp"][:19].replace("T", " "),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_model_info() -> None:
    """Model information page."""
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
            <h1 style="font-size:2rem; font-weight:800; margin-bottom:0.25rem;">
                ℹ️ Model Information
            </h1>
            <p style="color:#8b949e;">CNN + LSTM architecture details and configuration.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info = api_get("/model-info")
    if info is None:
        st.error("Cannot connect to backend.", icon="🔴")
        return

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(
            '<div class="section-header">🏗️ Architecture</div>', unsafe_allow_html=True
        )
        st.markdown(
            f"""
            | Property        | Value                       |
            |-----------------|-----------------------------|
            | Architecture    | `{info['architecture']}`    |
            | Input Shape     | `{info['input_shape']}`     |
            | Output Classes  | `{info['num_classes']}`     |
            | CNN Filters     | `[64, 128]`                 |
            | CNN Kernel Size | `3`                         |
            | LSTM Units      | `128`                       |
            | Dense Units     | `64`                        |
            | Dropout         | `0.4`                       |
            | Optimizer       | `Adam`                      |
            | Loss Function   | `Categorical Crossentropy`  |
            """,
        )

    with col2:
        st.markdown(
            '<div class="section-header">🏷️ Activity Classes</div>',
            unsafe_allow_html=True,
        )
        for i, lbl in enumerate(info["activity_labels"]):
            icon = ACTIVITY_ICONS.get(lbl, "🤸")
            color = ACTIVITY_COLORS.get(lbl, "#00d4ff")
            st.markdown(
                f"""
                <div style="background:#161b27; border:1px solid #21262d; border-radius:8px;
                            padding:0.6rem 1rem; margin-bottom:0.4rem; display:flex;
                            align-items:center; gap:0.75rem;">
                    <span style="font-size:1.3rem;">{icon}</span>
                    <span style="font-weight:600; color:{color};">
                        Class {i} — {lbl}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Model config
    cfg = info.get("config")
    if cfg:
        st.divider()
        st.markdown(
            '<div class="section-header">⚙️ Full Configuration</div>',
            unsafe_allow_html=True,
        )
        import json as _json

        st.code(_json.dumps(cfg, indent=2), language="json")

    # Architecture diagram
    st.divider()
    st.markdown(
        '<div class="section-header">🔄 Model Architecture Diagram</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
        ```
        Input (128, 9)
            │
            ▼
        Conv1D (64 filters, kernel=3, same padding)
        BatchNormalization → ReLU
        MaxPooling1D (pool_size=2)
        Dropout (0.4)
            │
            ▼
        Conv1D (128 filters, kernel=3, same padding)
        BatchNormalization → ReLU
        MaxPooling1D (pool_size=2)
        Dropout (0.4)
            │
            ▼
        LSTM (128 units)
        Dropout (0.4)
            │
            ▼
        Dense (64 units, ReLU)
        Dropout (0.2)
            │
            ▼
        Dense (6 units)
        Softmax → [WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS,
                   SITTING, STANDING, LAYING]
        ```
        """)

    # Dataset info
    st.divider()
    st.markdown('<div class="section-header">📂 Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
        **UCI HAR Dataset** — raw inertial signals (NOT engineered features)

        | Property      | Value                             |
        |---------------|-----------------------------------|
        | Training set  | 7,352 samples                     |
        | Test set      | 2,947 samples                     |
        | Window size   | 128 time steps                    |
        | Channels      | 9 (body_acc/gyro + total_acc)     |
        | Sampling rate | 50 Hz                             |
        | Subjects      | 30 volunteers (70/30 train/test)  |
        """)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    health = check_health()
    page = render_sidebar(health)

    if "🏠" in page:
        page_home(health)
    elif "📊" in page:
        page_analytics()
    elif "ℹ️" in page:
        page_model_info()


if __name__ == "__main__":
    main()
