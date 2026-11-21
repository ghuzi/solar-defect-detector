from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "models" / "best.pt"

BG, SURFACE = "#0B1F33", "#12304A"
TEXT, MUTED = "#EAF2F8", "#9FB4C7"
TEAL, AMBER, CORAL = "#1FA99B", "#F4A62A", "#E5604D"

# label, advice, accent colour, severity (higher = worse)
CLASS_INFO = {
    "clean": ("Clean", "No action needed.", TEAL, 0),
    "dust": ("Dusty", "Schedule a cleaning.", AMBER, 1),
    "bird_drops": ("Bird droppings", "Clean soon; droppings can cause hot spots.", AMBER, 2),
    "cracked": ("Cracked", "Inspect and repair or replace the panel.", CORAL, 3),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: __BG__; color: __TEXT__; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 3rem; max-width: 1150px; }

div.hero {
    background: linear-gradient(115deg, __SURFACE__ 0%, __TEAL__ 75%, __AMBER__ 100%);
    border-radius: 20px; padding: 2.4rem 2.6rem; margin-bottom: 1.6rem;
}
div.hero p.hero-title {
    font-family: 'Fraunces', serif !important; font-size: 2.6rem !important;
    line-height: 1.1 !important; color: #fff !important; margin: 0 !important;
}
div.hero p.hero-text {
    color: rgba(255,255,255,.92) !important; max-width: 640px;
    margin: .8rem 0 0 !important; font-size: 1.02rem !important;
}

.section-label {
    font-size: .78rem; letter-spacing: .12em; text-transform: uppercase;
    color: __TEAL__; font-weight: 700; margin: 1.6rem 0 .5rem;
}
.verdict {
    background: __SURFACE__; border: 1px solid rgba(255,255,255,.08);
    border-left: 6px solid var(--accent); border-radius: 14px;
    padding: 1rem 1.3rem; margin-bottom: 1rem;
}
.verdict-title { font-family: 'Fraunces', serif; font-size: 1.4rem; color: __TEXT__; margin: 0; }
.verdict-text { color: __MUTED__; margin: .25rem 0 0; }

[data-testid="stSidebar"] { background: __SURFACE__; border-right: 1px solid rgba(255,255,255,.06); }
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] label p { color: __TEXT__; }
[data-testid="stCaptionContainer"] p { color: __MUTED__ !important; }

[data-testid="stFileUploader"] section, [data-testid="stFileUploaderDropzone"] {
    background: __SURFACE__; border: 1.5px dashed __TEAL__; border-radius: 14px;
}
[data-testid="stFileUploader"] small { color: __MUTED__ !important; }
[data-testid="stFileUploader"] button { background: transparent; border: 1px solid __TEAL__; color: __TEXT__; }

[data-testid="stAlert"] {
    background: rgba(31,169,155,.12); border: 1px solid rgba(31,169,155,.35); border-radius: 14px;
}
[data-testid="stAlert"] * { color: __TEXT__ !important; }
[data-testid="stImage"] img { border-radius: 14px; }
</style>
"""
for token, value in {
    "__BG__": BG, "__SURFACE__": SURFACE, "__TEXT__": TEXT,
    "__MUTED__": MUTED, "__TEAL__": TEAL, "__AMBER__": AMBER,
}.items():
    CSS = CSS.replace(token, value)

st.set_page_config(page_title="Solar Panel Defect Detector", page_icon="☀️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


model = load_model()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### Detection settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    st.caption(
        "Lower values catch more objects but add false alarms. "
        "Bird droppings are the hardest class for this model, so try a lower threshold for them."
    )
    st.markdown("---")
    st.caption("Model: YOLOv8n fine-tuned on 6.5k solar panel images. Test mAP50: 0.70.")

# ---------- Hero ----------
st.markdown(
    """
    <div class="hero">
        <p class="hero-title">Solar Panel Defect Detector</p>
        <p class="hero-text">Upload a photo of a solar panel and the model tells you whether it is
        clean, dusty, covered in bird droppings or cracked, so maintenance teams know
        which panels need cleaning and which need replacing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="section-label">1. Upload</p>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded is None:
    st.info("Upload a JPG or PNG image to start.")
else:
    image = Image.open(uploaded).convert("RGB")
    result = model.predict(image, conf=conf, verbose=False)[0]

    st.markdown('<p class="section-label">2. Result</p>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        st.image(image, caption="Original", use_container_width=True)
    with right:
        st.image(result.plot()[..., ::-1], caption="Detections", use_container_width=True)

    rows = []
    for box in result.boxes:
        name = model.names[int(box.cls[0])]
        _, advice, _, _ = CLASS_INFO.get(name, (name, "", TEAL, 0))
        rows.append(
            {
                "Class": name,
                "Confidence": round(float(box.conf[0]), 3),
                "Suggested action": advice,
            }
        )

    st.markdown('<p class="section-label">3. Findings</p>', unsafe_allow_html=True)
    if not rows:
        st.warning("Nothing detected. Try lowering the confidence threshold.")
    else:
        df = pd.DataFrame(rows).sort_values("Confidence", ascending=False)
        worst = max(df["Class"], key=lambda c: CLASS_INFO.get(c, ("", "", TEAL, 0))[3])
        label, advice, accent, _ = CLASS_INFO[worst]
        st.markdown(
            f"""
            <div class="verdict" style="--accent: {accent}">
                <p class="verdict-title">{label}</p>
                <p class="verdict-text">{advice}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                )
            },
        )
        st.caption(f"{len(df)} object(s) detected at confidence ≥ {conf:.2f}")