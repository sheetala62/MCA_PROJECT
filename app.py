import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageStat
import plotly.graph_objects as go
import pandas as pd
import io
import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_COLORS  = ["#dc2626", "#ea580c", "#16a34a", "#2563eb"]
CLASS_ICONS   = ["🔴", "🟠", "🟢", "🔵"]

CLASS_DESC = {
    "Glioma":     "Originates in glial cells of the brain or spine. Most common malignant brain tumor. Requires urgent clinical evaluation.",
    "Meningioma": "Arises from the meninges surrounding the brain. Usually benign but can cause pressure on the brain.",
    "No Tumor":   "No tumor detected. MRI scan appears within normal range.",
    "Pituitary":  "Forms in the pituitary gland at the base of the brain. Often benign and highly treatable.",
}

CLASS_RISK = {
    "Glioma":     ("High Risk", "#dc2626", "Immediate neurological consultation recommended."),
    "Meningioma": ("Moderate Risk", "#ea580c", "Follow-up MRI and specialist evaluation advised."),
    "No Tumor":   ("No Risk", "#16a34a", "No immediate action required. Routine follow-up."),
    "Pituitary":  ("Low–Moderate Risk", "#2563eb", "Endocrinology or neurosurgery consultation advised."),
}

CLASS_CLINICAL = {
    "Glioma": {
        "Symptoms": "Headaches, seizures, cognitive changes, motor weakness",
        "Treatment": "Surgery, radiation therapy, chemotherapy (temozolomide)",
        "Survival":  "Varies by grade: Grade I–II (years to decades), Grade III–IV (months to 2 years)",
        "Prevalence":"Most common malignant primary brain tumor (~80% of cases)",
    },
    "Meningioma": {
        "Symptoms": "Headaches, vision problems, hearing loss, memory difficulty",
        "Treatment": "Observation, surgery, or stereotactic radiosurgery",
        "Survival":  "Excellent for benign (Grade I); 10-year survival >80%",
        "Prevalence":"Most common benign brain tumor; 2× more common in women",
    },
    "No Tumor": {
        "Symptoms": "No tumor-related symptoms expected",
        "Treatment": "No tumor-specific treatment required",
        "Survival":  "Normal life expectancy",
        "Prevalence":"Normal brain MRI finding",
    },
    "Pituitary": {
        "Symptoms": "Vision problems, hormonal imbalances, headaches, fatigue",
        "Treatment": "Medication (dopamine agonists), surgery, or radiation",
        "Survival":  "Excellent; most pituitary adenomas are benign",
        "Prevalence":"Occurs in ~10–15% of all primary brain tumors",
    },
}

BASELINE_ACC = 0.9388
DUAL_ACC     = 0.9425

# ── Session state for history ─────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── CSS — Clean Professional White/Light Theme ────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #f8fafc !important;
    color: #1e293b !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebar"] * { color: #1e293b !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #0f172a !important; }

/* ── Sidebar collapse/expand toggle button — always visible ── */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] *,
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] *,
button[kind="header"],
button[kind="header"] * {
    color: #0f172a !important;
    background: transparent !important;
    opacity: 1 !important;
    visibility: visible !important;
}
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
/* Ensure the > arrow button that brings sidebar back is always shown */
section[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    background: #f8fafc !important;
    border-right: 1px solid #e2e8f0 !important;
}
section[data-testid="stSidebarCollapsedControl"] * {
    color: #0f172a !important;
    visibility: visible !important;
}

/* ── Header strip ── */
.top-bar {
    background: #0f172a;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}
.top-bar-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
}
.top-bar-sub {
    font-size: 0.88rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}
.top-bar-badges { display: flex; gap: 0.6rem; flex-wrap: wrap; }
.badge {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.75rem;
    color: #cbd5e1;
    font-weight: 500;
}

/* ── Cards ── */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.card-title {
    font-size: 0.72rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.6rem;
}

/* ── Section title ── */
.sec-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0f172a;
    margin: 1.6rem 0 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #e2e8f0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Stat chips ── */
.stat-row { display: flex; gap: 0.8rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.stat-chip {
    flex: 1; min-width: 130px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.sc-val  { font-size: 1.5rem; font-weight: 800; color: #0f172a; }
.sc-lbl  { font-size: 0.7rem; color: #94a3b8; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Result cards ── */
.result-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    border-radius: 14px 14px 0 0;
}
.result-card.baseline::before { background: #3b82f6; }
.result-card.dual::before     { background: #f59e0b; }
.model-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
.model-label.baseline { color: #3b82f6; }
.model-label.dual     { color: #d97706; }
.pred-class  { font-size: 1.5rem; font-weight: 800; margin: 0.2rem 0 0.5rem; }
.conf-bar-wrap { background: #f1f5f9; border-radius: 8px; height: 8px; overflow: hidden; margin: 0.5rem 0; }
.conf-bar      { height: 100%; border-radius: 8px; }
.conf-text     { font-size: 0.82rem; color: #64748b; margin-top: 0.3rem; }
.desc-text     { font-size: 0.80rem; color: #94a3b8; margin-top: 0.5rem; line-height: 1.5; }

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    border-radius: 6px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 700;
    margin-top: 0.4rem;
    letter-spacing: 0.04em;
}

/* ── Agreement banner ── */
.agree-banner {
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-bottom: 1rem;
    font-weight: 600;
    font-size: 0.88rem;
}
.agree-banner.yes { background: #f0fdf4; border: 1px solid #86efac; color: #15803d; }
.agree-banner.no  { background: #fef2f2; border: 1px solid #fca5a5; color: #b91c1c; }

/* ── Placeholder ── */
.placeholder {
    background: #ffffff;
    border: 2px dashed #cbd5e1;
    border-radius: 14px;
    padding: 3.5rem 2rem;
    text-align: center;
    margin: 0.5rem 0 1.5rem;
}
.placeholder h3 { font-size: 1.2rem !important; font-weight: 700 !important; color: #0f172a !important; margin-bottom: 0.5rem !important; }
.placeholder p  { color: #94a3b8; font-size: 0.9rem; }

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] * { color: #64748b !important; }

/* ── Sidebar cards ── */
.sb-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.7rem;
}
.sb-card-label { font-size: 0.7rem; color: #94a3b8; margin-bottom: 0.2rem; font-weight: 500; text-transform: uppercase; }
.sb-card-value { font-size: 1.5rem; font-weight: 800; }
.sb-card-sub   { font-size: 0.68rem; color: #94a3b8; margin-top: 0.1rem; }

/* ── Clinical info table ── */
.clin-row { display: flex; gap: 0.5rem; margin-bottom: 0.4rem; font-size: 0.83rem; }
.clin-key { font-weight: 600; color: #475569; min-width: 100px; }
.clin-val { color: #64748b; }

/* ── History item ── */
.hist-item {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.4rem;
    font-size: 0.8rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* ── Image quality pill ── */
.quality-pill {
    display: inline-block;
    border-radius: 20px;
    padding: 0.15rem 0.6rem;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 0.4rem;
}

/* hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import DenseNet121

    def channel_attention(x, ratio=16):
        channel = x.shape[-1]
        avg_pool = layers.GlobalAveragePooling2D()(x)
        avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
        d1 = layers.Dense(channel // ratio, activation="relu")(avg_pool)
        d2 = layers.Dense(channel, activation="sigmoid")(d1)
        return layers.Multiply()([x, d2])

    def spatial_attention(x):
        avg = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
        mx  = layers.Lambda(lambda t: tf.reduce_max(t,  axis=-1, keepdims=True))(x)
        cat = layers.Concatenate(axis=-1)([avg, mx])
        att = layers.Conv2D(1, 7, padding="same", activation="sigmoid")(cat)
        return layers.Multiply()([x, att])

    def build_baseline():
        base = DenseNet121(weights=None, include_top=False, input_shape=(224,224,3))
        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        return models.Model(base.input, layers.Dense(4, activation="softmax")(x))

    def build_dual():
        base = DenseNet121(weights=None, include_top=False, input_shape=(224,224,3))
        x = channel_attention(base.output)
        x = spatial_attention(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        return models.Model(base.input, layers.Dense(4, activation="softmax")(x))

    b = build_baseline(); b.load_weights("baseline_densenet.h5",      by_name=False)
    d = build_dual();     d.load_weights("dual_attention_densenet.h5", by_name=False)
    return b, d


# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def predict(model, arr):
    probs = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]) * 100, probs

def assess_image_quality(img: Image.Image):
    """Check brightness and contrast of uploaded image."""
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    mean = stat.mean[0]
    std  = stat.stddev[0]
    if mean < 20:
        return "Too Dark", "#dc2626"
    elif mean > 235:
        return "Too Bright", "#ea580c"
    elif std < 15:
        return "Low Contrast", "#d97706"
    else:
        return "Good Quality", "#16a34a"

def conf_chart(probs, title, accent):
    pred_idx = int(np.argmax(probs))
    colors = [accent if i == pred_idx else "#e2e8f0" for i in range(4)]
    fig = go.Figure(go.Bar(
        y=CLASS_NAMES, x=[p * 100 for p in probs], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"<b>{p*100:.1f}%</b>" if i == pred_idx else f"{p*100:.1f}%"
              for i, p in enumerate(probs)],
        textposition="outside",
        textfont=dict(color="#1e293b", size=12, family="Inter"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#475569", size=13, family="Inter"), x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e293b", family="Inter"),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", showticklabels=False, range=[0, 125]),
        yaxis=dict(showgrid=False, tickfont=dict(size=13, color="#475569")),
        margin=dict(l=5, r=10, t=35, b=5), height=210,
    )
    return fig

def compare_chart(b_probs, d_probs):
    fig = go.Figure()
    for name, probs, color in [
        ("Baseline DenseNet121",       b_probs, "#3b82f6"),
        ("Dual Attention DenseNet121", d_probs, "#f59e0b"),
    ]:
        fig.add_trace(go.Bar(
            name=name, x=CLASS_NAMES, y=[p*100 for p in probs],
            marker=dict(color=color, opacity=0.85, line=dict(width=0)),
            text=[f"{p*100:.1f}%" for p in probs], textposition="outside",
            textfont=dict(color="#1e293b", size=11),
        ))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e293b", family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, tickfont=dict(size=13, color="#475569")),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Confidence (%)",
                   range=[0, 125], tickfont=dict(size=11)),
        margin=dict(l=5, r=5, t=40, b=5), height=280,
    )
    return fig

def acc_chart():
    fig = go.Figure()
    for name, acc, color in [
        ("Baseline\nDenseNet121",        BASELINE_ACC, "#3b82f6"),
        ("Dual Attention\nDenseNet121",  DUAL_ACC,     "#f59e0b"),
    ]:
        fig.add_trace(go.Bar(
            x=[name], y=[acc * 100],
            marker=dict(color=color, opacity=0.9, line=dict(width=0)),
            text=[f"<b>{acc*100:.2f}%</b>"], textposition="outside",
            textfont=dict(color="#1e293b", size=15), width=0.4,
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e293b", family="Inter"),
        showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(size=13, color="#475569")),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Accuracy (%)",
                   range=[90, 97], tickfont=dict(size=11)),
        margin=dict(l=5, r=5, t=20, b=5), height=300,
    )
    return fig

def generate_report(filename, b_class, b_conf, b_probs,
                    d_class, d_conf, d_probs, quality):
    """Generate a plain-text downloadable report."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 60,
        "       NEUROSCAN AI — BRAIN TUMOR CLASSIFICATION REPORT",
        "=" * 60,
        f"  File         : {filename}",
        f"  Timestamp    : {ts}",
        f"  Image Quality: {quality}",
        "-" * 60,
        "  MODEL RESULTS",
        "-" * 60,
        f"  Baseline DenseNet121",
        f"    Prediction : {b_class}",
        f"    Confidence : {b_conf:.2f}%",
        f"    Model Acc. : {BASELINE_ACC*100:.2f}%",
        "",
        f"  Dual Attention DenseNet121  [BEST MODEL]",
        f"    Prediction : {d_class}",
        f"    Confidence : {d_conf:.2f}%",
        f"    Model Acc. : {DUAL_ACC*100:.2f}%",
        "-" * 60,
        "  PROBABILITY BREAKDOWN",
        "-" * 60,
    ]
    for cls, b, d in zip(CLASS_NAMES, b_probs, d_probs):
        lines.append(f"  {cls:<14}: Baseline {b*100:>6.2f}%  |  Dual Attn {d*100:>6.2f}%")
    lines += [
        "-" * 60,
        "  CLINICAL NOTE",
        "-" * 60,
        f"  Risk Level   : {CLASS_RISK[d_class][0]}",
        f"  Advisory     : {CLASS_RISK[d_class][2]}",
        "-" * 60,
        "  DISCLAIMER: This report is generated by an AI system for",
        "  research and educational purposes only. It does NOT",
        "  constitute a medical diagnosis. Always consult a qualified",
        "  medical professional for clinical decisions.",
        "=" * 60,
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:0.8rem 0 0.5rem; border-bottom:1px solid #e2e8f0; margin-bottom:1rem;">
        <div style="font-size:1.1rem; font-weight:800; color:#0f172a;">🧠 NeuroScan AI</div>
        <div style="font-size:0.72rem; color:#94a3b8; margin-top:0.2rem;">Brain Tumor Classifier</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.7rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.6rem;'>Model Performance</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-card" style="border-left:3px solid #3b82f6;">
        <div class="sb-card-label">Baseline DenseNet121</div>
        <div class="sb-card-value" style="color:#3b82f6;">{BASELINE_ACC*100:.2f}%</div>
        <div class="sb-card-sub">Test Set Accuracy</div>
    </div>
    <div class="sb-card" style="border-left:3px solid #f59e0b;">
        <div class="sb-card-label">Dual Attention DenseNet121 ✦ Best</div>
        <div class="sb-card-value" style="color:#d97706;">{DUAL_ACC*100:.2f}%</div>
        <div class="sb-card-sub">Test Set Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.7rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin:0.8rem 0 0.6rem;'>Tumor Classes</div>", unsafe_allow_html=True)
    for name, color, icon in zip(CLASS_NAMES, CLASS_COLORS, CLASS_ICONS):
        st.markdown(f"""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-left:3px solid {color};
             border-radius:8px; padding:0.55rem 0.8rem; margin-bottom:0.4rem;">
            <div style="font-size:0.82rem; font-weight:600; color:{color};">{icon} {name}</div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:0.15rem; line-height:1.4;">{CLASS_DESC[name]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.7rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin:0.8rem 0 0.6rem;'>Architecture</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:0.7rem 0.9rem; font-size:0.78rem; color:#64748b; line-height:2;">
        <b style="color:#475569;">Backbone</b>: DenseNet121<br>
        <b style="color:#475569;">Pretrained</b>: ImageNet<br>
        <b style="color:#475569;">Input</b>: 224 × 224 × 3<br>
        <b style="color:#475569;">Classes</b>: 4<br>
        <b style="color:#475569;">Attention</b>: Channel + Spatial<br>
        <b style="color:#475569;">Dataset</b>: Kaggle Brain MRI (7,023 images)
    </div>
    """, unsafe_allow_html=True)

    # Session history in sidebar
    if st.session_state.history:
        st.markdown("<div style='font-size:0.7rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin:0.8rem 0 0.6rem;'>Session History</div>", unsafe_allow_html=True)
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            st.markdown(f"""
            <div class="hist-item">
                <span style="color:#475569;">{h['file'][:18]}…</span>
                <span style="color:{CLASS_COLORS[CLASS_NAMES.index(h['pred'])]}; font-weight:600;">{h['pred']}</span>
            </div>
            """, unsafe_allow_html=True)
        if st.button("🗑 Clear History", width='stretch'):
            st.session_state.history = []
            st.rerun()

    st.markdown("<div style='font-size:0.68rem; color:#cbd5e1; margin-top:1.5rem; line-height:1.6; padding:0.6rem; background:#f8fafc; border-radius:8px; border:1px solid #e2e8f0;'>⚠️ <b>Disclaimer:</b> This tool is for research and educational purposes only. Not a substitute for professional medical diagnosis.</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="top-bar">
    <div>
        <div class="top-bar-title">🧠 NeuroScan AI</div>
        <div class="top-bar-sub">Brain Tumor Classification using Dual-Attention DenseNet121m</div>
    </div>
    <div class="top-bar-badges">
        <span class="badge">🎯 94.25% Accuracy</span>
        <span class="badge">⚡ Real-time Inference</span>
        <span class="badge">🔬 4 Tumor Classes</span>
        <span class="badge">🤖 Dual Attention DenseNet121</span>
        <span class="badge">📊 Kaggle Dataset · 7,023 MRI</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
with st.spinner("Loading neural network models…"):
    try:
        baseline_model, dual_model = load_models()
        st.success("✅ Both models loaded successfully and ready for inference.")
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        st.stop()

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-title">📤 Upload MRI Scan</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop a brain MRI image here (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    help="Upload a T1-weighted brain MRI scan in JPG or PNG format for automated classification.",
)

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if uploaded:
    image = Image.open(uploaded)
    arr   = preprocess(image)

    # Image quality check
    quality_label, quality_color = assess_image_quality(image)

    with st.spinner("Running inference on both models…"):
        b_class, b_conf, b_probs = predict(baseline_model, arr)
        d_class, d_conf, d_probs = predict(dual_model, arr)

    # Save to session history
    st.session_state.history.append({
        "file": uploaded.name,
        "pred": d_class,
        "conf": d_conf,
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
    })

    b_color = CLASS_COLORS[CLASS_NAMES.index(b_class)]
    d_color = CLASS_COLORS[CLASS_NAMES.index(d_class)]
    agree   = b_class == d_class
    risk_label, risk_color, risk_advice = CLASS_RISK[d_class]

    # ── Stat chips ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">📊 Analysis Overview</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-chip">
            <div class="sc-val" style="color:#3b82f6;">{b_conf:.1f}%</div>
            <div class="sc-lbl">Baseline Confidence</div>
        </div>
        <div class="stat-chip">
            <div class="sc-val" style="color:#d97706;">{d_conf:.1f}%</div>
            <div class="sc-lbl">Dual Attn Confidence</div>
        </div>
        <div class="stat-chip">
            <div class="sc-val" style="color:{'#16a34a' if agree else '#dc2626'};">{'✓' if agree else '✗'}</div>
            <div class="sc-lbl">Models {'Agree' if agree else 'Disagree'}</div>
        </div>
        <div class="stat-chip">
            <div class="sc-val" style="color:{risk_color};">{risk_label}</div>
            <div class="sc-lbl">Risk Level</div>
        </div>
        <div class="stat-chip">
            <div class="sc-val" style="color:{quality_color}; font-size:1rem;">{quality_label}</div>
            <div class="sc-lbl">Image Quality</div>
        </div>
        <div class="stat-chip">
            <div class="sc-val" style="color:#7c3aed;">{d_conf - b_conf:+.1f}%</div>
            <div class="sc-lbl">Confidence Delta</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Image + Result cards ────────────────────────────────────────────────
    img_col, res_col = st.columns([1, 2], gap="large")

    with img_col:
        st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded MRI Scan", width='stretch')
        st.markdown(f"""
        <div style="margin-top:0.8rem; font-size:0.78rem; color:#94a3b8; text-align:center; line-height:1.8;">
            <b>File:</b> {uploaded.name}<br>
            <b>Size:</b> {image.size[0]} × {image.size[1]} px<br>
            <b>Quality:</b> <span style="color:{quality_color}; font-weight:600;">{quality_label}</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with res_col:
        agree_cls  = "yes" if agree else "no"
        agree_icon = "✅" if agree else "⚠️"
        agree_msg  = "Both models agree on the prediction." if agree else "Models have different predictions — manual review recommended."
        st.markdown(f'<div class="agree-banner {agree_cls}">{agree_icon} {agree_msg}</div>', unsafe_allow_html=True)

        # Baseline card
        st.markdown(f"""
        <div class="result-card baseline">
            <div class="model-label baseline">Baseline DenseNet121</div>
            <div class="pred-class" style="color:{b_color};">{b_class}</div>
            <div class="conf-bar-wrap">
                <div class="conf-bar" style="width:{b_conf:.1f}%; background:#3b82f6;"></div>
            </div>
            <div class="conf-text">Confidence: <strong style="color:{b_color};">{b_conf:.2f}%</strong>
                &nbsp;·&nbsp; Test Accuracy: <strong style="color:#3b82f6;">{BASELINE_ACC*100:.2f}%</strong>
            </div>
            <div class="desc-text">{CLASS_DESC[b_class]}</div>
        </div>
        """, unsafe_allow_html=True)

        # Dual Attention card
        st.markdown(f"""
        <div class="result-card dual">
            <div class="model-label dual">Dual Attention DenseNet121
                <span style="background:#f59e0b; color:#ffffff; font-size:0.65rem;
                      padding:0.1rem 0.5rem; border-radius:8px; margin-left:0.4rem; font-weight:700;">
                    ✦ BEST MODEL
                </span>
            </div>
            <div class="pred-class" style="color:{d_color};">{d_class}</div>
            <div class="conf-bar-wrap">
                <div class="conf-bar" style="width:{d_conf:.1f}%; background:#f59e0b;"></div>
            </div>
            <div class="conf-text">Confidence: <strong style="color:{d_color};">{d_conf:.2f}%</strong>
                &nbsp;·&nbsp; Test Accuracy: <strong style="color:#d97706;">{DUAL_ACC*100:.2f}%</strong>
            </div>
            <div class="desc-text">{CLASS_DESC[d_class]}</div>
            <div class="risk-badge" style="background:{risk_color}20; color:{risk_color}; border:1px solid {risk_color}50;">
                ⚡ {risk_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Clinical Advisory ───────────────────────────────────────────────────
    st.markdown('<div class="sec-title">🏥 Clinical Information</div>', unsafe_allow_html=True)
    clin = CLASS_CLINICAL[d_class]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Risk Assessment — {d_class}</div>
            <div style="font-size:1.1rem; font-weight:700; color:{risk_color}; margin-bottom:0.5rem;">{risk_label}</div>
            <div style="font-size:0.83rem; color:#64748b; line-height:1.6;">{risk_advice}</div>
            <div style="margin-top:0.8rem; padding-top:0.8rem; border-top:1px solid #f1f5f9;">
                <div class="clin-row"><span class="clin-key">Symptoms</span><span class="clin-val">{clin['Symptoms']}</span></div>
                <div class="clin-row"><span class="clin-key">Prevalence</span><span class="clin-val">{clin['Prevalence']}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Treatment & Prognosis</div>
            <div class="clin-row"><span class="clin-key">Treatment</span><span class="clin-val">{clin['Treatment']}</span></div>
            <div class="clin-row" style="margin-top:0.5rem;"><span class="clin-key">Survival</span><span class="clin-val">{clin['Survival']}</span></div>
            <div style="margin-top:0.8rem; padding:0.6rem 0.8rem; background:#fef9c3; border-radius:8px; border:1px solid #fde68a; font-size:0.78rem; color:#92400e;">
                ⚠️ This information is for educational purposes only. Consult a qualified neurologist or oncologist for diagnosis and treatment decisions.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Confidence Charts ───────────────────────────────────────────────────
    st.markdown('<div class="sec-title">📈 Confidence Breakdown</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(conf_chart(b_probs, "Baseline DenseNet121", "#3b82f6"), width='stretch', key="ch_b")
    with c2:
        st.plotly_chart(conf_chart(d_probs, "Dual Attention DenseNet121", "#f59e0b"), width='stretch', key="ch_d")

    # ── Comparison Chart ────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">⚖️ Side-by-Side Model Comparison</div>', unsafe_allow_html=True)
    st.plotly_chart(compare_chart(b_probs, d_probs), width='stretch', key="ch_cmp")

    # ── Probability Table ───────────────────────────────────────────────────
    st.markdown('<div class="sec-title">📋 Detailed Probability Table</div>', unsafe_allow_html=True)
    rows = [{
        "Class":                  cls,
        "Baseline %":             f"{b*100:.2f}%",
        "Dual Attention %":       f"{d*100:.2f}%",
        "Δ (Dual − Baseline)":   f"{(d-b)*100:+.2f}%",
        "Predicted by Baseline":  "✅" if CLASS_NAMES.index(cls) == int(np.argmax(b_probs)) else "—",
        "Predicted by Dual Attn": "✅" if CLASS_NAMES.index(cls) == int(np.argmax(d_probs)) else "—",
    } for cls, b, d in zip(CLASS_NAMES, b_probs, d_probs)]
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    # ── Download Report ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">📥 Download Report</div>', unsafe_allow_html=True)
    report_text = generate_report(
        uploaded.name, b_class, b_conf, b_probs,
        d_class, d_conf, d_probs, quality_label
    )
    col_dl, col_info = st.columns([1, 3], gap="large")
    with col_dl:
        st.download_button(
            label="⬇️ Download Classification Report (.txt)",
            data=report_text,
            file_name=f"neuroscan_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            width='stretch',
        )
    with col_info:
        st.markdown("""
        <div style="font-size:0.82rem; color:#94a3b8; padding:0.6rem 0; line-height:1.7;">
            The report includes: file name, timestamp, image quality, predictions from both models,
            full probability breakdown, risk level, and clinical advisory note.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PLACEHOLDER — no image uploaded
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div class="placeholder">
        <div style="font-size:3rem; margin-bottom:0.8rem;">🧠</div>
        <h3>Ready for Analysis</h3>
        <p>Upload a brain MRI scan above to get instant classification<br>
           from both the Baseline and Dual Attention DenseNet121 models.</p>
    </div>
    """, unsafe_allow_html=True)

    # Overall accuracy chart
    st.markdown('<div class="sec-title">📊 Overall Test Set Accuracy Comparison</div>', unsafe_allow_html=True)
    st.plotly_chart(acc_chart(), width='stretch', key="ch_acc")

    # How it works
    st.markdown('<div class="sec-title">🔬 How It Works</div>', unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4, gap="large")
    steps = [
        ("📤", "1. Upload", "Upload a brain MRI scan in JPG or PNG format."),
        ("⚙️", "2. Preprocess", "Image resized to 224×224 px and normalized to [0,1]."),
        ("🤖", "3. Classify", "Both DenseNet121 models run inference simultaneously."),
        ("📊", "4. Compare", "View predictions, confidence scores, risk level, and downloadable report."),
    ]
    for col, (icon, title, body) in zip([h1, h2, h3, h4], steps):
        with col:
            col.markdown(f"""
            <div class="card" style="text-align:center; padding:1.6rem 1.2rem;">
                <div style="font-size:2rem; margin-bottom:0.7rem;">{icon}</div>
                <div style="font-size:0.92rem; font-weight:700; color:#0f172a; margin-bottom:0.4rem;">{title}</div>
                <div style="font-size:0.8rem; color:#94a3b8; line-height:1.6;">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    # Tumor info reference cards
    st.markdown('<div class="sec-title">📚 Tumor Class Reference</div>', unsafe_allow_html=True)
    cols = st.columns(4, gap="large")
    for col, (name, color, icon) in zip(cols, zip(CLASS_NAMES, CLASS_COLORS, CLASS_ICONS)):
        risk_l, risk_c, _ = CLASS_RISK[name]
        with col:
            col.markdown(f"""
            <div class="card">
                <div style="font-size:1.5rem; margin-bottom:0.4rem;">{icon}</div>
                <div style="font-size:0.95rem; font-weight:700; color:{color}; margin-bottom:0.3rem;">{name}</div>
                <div style="font-size:0.75rem; color:#94a3b8; line-height:1.5; margin-bottom:0.5rem;">{CLASS_DESC[name]}</div>
                <div style="font-size:0.72rem; font-weight:600; color:{risk_c}; background:{risk_c}15;
                     border-radius:5px; padding:0.15rem 0.5rem; display:inline-block;">{risk_l}</div>
            </div>
            """, unsafe_allow_html=True)