import html as _html
import io

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from engine import ParkVisionEngine

st.set_page_config(
    page_title="ParkVision AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Theme tokens (Sakura & White)
# ──────────────────────────────────────────────────────────────────────────────
VACANT_RGB = (46, 196, 134)     # fresh leaf green
OCCUPIED_RGB = (232, 74, 127)   # deep sakura crimson

STATUS_HEX = {
    "green": "#2EC486",
    "blue": "#F472B6",
    "orange": "#F59E0B",
    "yellow": "#FBBF24",
    "red": "#E84A7F",
    "violet": "#D977A8",
    "gray": "#9E7B8B",
    "grey": "#9E7B8B",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Instrument+Sans:wght@400;500;600&display=swap');

:root{
  --night:#FFF9FB; --ink:#432333; --muted:#946B7E;
  --beam:#F472B6; --dusk:#FB92B8; --lagoon:#2EC486; --signal:#E84A7F; --amber:#F59E0B;
  --stroke:rgba(244,114,182,.26);
  --ease:cubic-bezier(.22,1,.36,1);
}

/* ── Base ─────────────────────────────────────────────── */
html, body, .stApp{ font-family:'Instrument Sans',system-ui,sans-serif; color:var(--ink); }
h1,h2,h3,.pv-title,.pv-h,.pv-value{ font-family:'Bricolage Grotesque',system-ui,sans-serif; letter-spacing:-.015em; color:var(--ink); }
#MainMenu, footer, .stDeployButton{ visibility:hidden; }
header[data-testid="stHeader"]{ background:transparent; }
.block-container{ padding-top:2.2rem; max-width:1180px; }
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"], [data-testid="stSidebar"] p{ color:var(--ink); }
[data-testid="stCaptionContainer"]{ color:var(--muted); }

/* ── Sakura & White Aurora background ─────────────────── */
.stApp{
  isolation:isolate;
  background:
    radial-gradient(1100px 700px at 8% -10%, #FFE3EE 0%, transparent 60%),
    radial-gradient(900px 600px at 100% 0%, #FFD6E7 0%, transparent 55%),
    var(--night);
}
.stApp::before, .stApp::after{
  content:""; position:fixed; z-index:-1; border-radius:50%;
  filter:blur(100px); opacity:.5; pointer-events:none;
}
.stApp::before{ width:520px; height:520px; left:-140px; top:22%;
  background:radial-gradient(circle,#FFB8D2,transparent 70%);
  animation:drift1 34s ease-in-out infinite alternate; }
.stApp::after{ width:620px; height:620px; right:-180px; bottom:-140px;
  background:radial-gradient(circle,#FFC8DF,transparent 70%);
  animation:drift2 40s ease-in-out infinite alternate; }
@keyframes drift1{ to{ transform:translate(160px,-110px) scale(1.18); } }
@keyframes drift2{ to{ transform:translate(-200px,-140px) scale(1.12); } }

/* ── Liquid glass surface ─────────────────────────────── */
.glass{
  position:relative; overflow:hidden; border-radius:24px;
  background:linear-gradient(140deg, rgba(255,255,255,.88), rgba(255,240,246,.65) 60%);
  backdrop-filter:blur(26px) saturate(180%); -webkit-backdrop-filter:blur(26px) saturate(180%);
  border:1px solid var(--stroke);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,.95),
    inset 0 -1px 0 rgba(244,114,182,.08),
    0 24px 50px -24px rgba(219,112,155,.22);
  transition:transform .5s var(--ease), box-shadow .5s var(--ease), border-color .5s var(--ease);
}
.glass::before{
  content:""; position:absolute; inset:0; pointer-events:none;
  background:linear-gradient(115deg, transparent 32%, rgba(255,255,255,.75) 47%, transparent 62%);
  transform:translateX(-130%); transition:transform 1.1s var(--ease);
}
.glass:hover{ border-color:rgba(244,114,182,.5); }
.glass:hover::before{ transform:translateX(130%); }

/* ── Sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg, rgba(255,255,255,.85), rgba(255,236,244,.75));
  backdrop-filter:blur(30px) saturate(180%); -webkit-backdrop-filter:blur(30px) saturate(180%);
  border-right:1px solid var(--stroke);
}
[data-testid="stSidebar"] > div:first-child{ background:transparent; }
.pv-side-title{ font:700 1.25rem 'Bricolage Grotesque',sans-serif; margin:.2rem 0 1rem; color:var(--ink); }
.pv-tip{ padding:14px 16px; font-size:.86rem; color:var(--muted); line-height:1.5; border-radius:18px; }

/* ── Sidebar version badge ────────────────────────────── */
.pv-badge{
  display:inline-flex; align-items:center; gap:6px; margin-top:1rem;
  padding:6px 14px; border-radius:99px; font-size:.78rem; font-weight:600;
  background:rgba(244,114,182,.1); color:var(--beam); border:1px solid rgba(244,114,182,.2);
}

/* ── Hero ─────────────────────────────────────────────── */
.pv-hero{ display:flex; align-items:center; gap:20px; padding:26px 30px; margin-bottom:1.4rem; }
.pv-mark{
  width:62px; height:62px; flex:none; display:grid; place-items:center;
  border-radius:20px; background:linear-gradient(145deg, rgba(255,182,210,.75), rgba(244,114,182,.55));
  box-shadow:inset 0 1px 0 rgba(255,255,255,.9), 0 10px 30px -8px rgba(244,114,182,.45);
}
.pv-mark svg{ width:32px; height:32px; }
.pv-title{ font-size:2.1rem; font-weight:700; line-height:1.1; color:var(--ink); }
.pv-sub{ color:var(--muted); margin-top:6px; font-size:1.02rem; max-width:60ch; }

/* ── Empty state ──────────────────────────────────────── */
.pv-steps{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; margin-top:1.1rem; }
.pv-step{ padding:22px 24px; transition:transform .35s var(--ease); }
.pv-step:hover{ transform:translateY(-3px); }
.pv-step-num{
  display:inline-flex; align-items:center; justify-content:center;
  width:36px; height:36px; border-radius:12px; font:700 1.1rem 'Bricolage Grotesque',sans-serif;
  background:linear-gradient(135deg, #FFD6E8, #F472B6); color:#fff;
  box-shadow:0 4px 12px -2px rgba(244,114,182,.4);
}
.pv-step span{ display:block; font-weight:600; margin-top:10px; color:var(--ink); }
.pv-step small{ display:block; color:var(--muted); margin-top:4px; line-height:1.45; }

/* ── File panel + chips ───────────────────────────────── */
.pv-panel{ padding:20px 22px; }
.pv-h{ font-weight:700; font-size:1.05rem; color:var(--ink); }
.pv-chips{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
.pv-chip{
  display:inline-flex; align-items:center; gap:8px; padding:6px 13px; border-radius:99px; font-size:.84rem;
  background:rgba(255,240,246,.85); border:1px solid var(--stroke); color:var(--ink);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.9);
}
.pv-chip.warn{ border-color:rgba(245,158,11,.55); color:#B45309; background:rgba(254,243,199,.7); }
.pv-dot{ width:8px; height:8px; border-radius:50%; background:var(--c,var(--beam)); animation:pulse 2.2s infinite; }
@keyframes pulse{ 0%{ box-shadow:0 0 0 0 var(--c,var(--beam)); } 70%,100%{ box-shadow:0 0 0 9px transparent; } }

/* ── Metrics (the one orchestrated reveal) ────────────── */
.pv-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:16px; margin:1.2rem 0; }
.pv-metric{ padding:20px 24px 20px 28px; }
.pv-metric::after{
  content:""; position:absolute; left:0; top:20px; bottom:20px; width:4px; border-radius:4px;
  background:var(--accent); box-shadow:0 0 14px var(--accent);
}
.pv-label{ color:var(--muted); font-size:.92rem; }
.pv-value{ font-size:2.7rem; font-weight:700; line-height:1.1; margin-top:6px; display:flex; align-items:baseline; color:var(--ink); }
.pv-value small{ font-size:1.3rem; color:var(--muted); font-weight:500; }
@property --n{ syntax:'<integer>'; inherits:false; initial-value:0; }
.count{ counter-reset:n var(--n); animation:countup 1.2s var(--ease) both; animation-delay:calc(var(--i,0)*110ms + 150ms); }
.count::after{ content:counter(n); }
@keyframes countup{ from{ --n:0; } to{ --n:var(--target); } }
.reveal{ animation:rise .9s var(--ease) both; animation-delay:calc(var(--i,0)*110ms); }
@keyframes rise{ from{ opacity:0; transform:translateY(16px); } to{ opacity:1; transform:none; } }

.pv-split{ display:grid; grid-template-columns:1.2fr 1fr; gap:16px; margin-bottom:1.6rem; }
@media (max-width:820px){ .pv-split{ grid-template-columns:1fr; } .pv-hero{ padding:20px; } }
.pv-row{ display:flex; justify-content:space-between; align-items:center; gap:12px; }
.pv-bar{ height:14px; margin-top:18px; border-radius:99px; background:rgba(244,114,182,.14); overflow:hidden;
  box-shadow:inset 0 2px 4px rgba(180,90,130,.15); }
.pv-bar > span{
  display:block; height:100%; width:var(--w); border-radius:inherit;
  background:linear-gradient(90deg, var(--c), color-mix(in srgb, var(--c) 65%, white));
  box-shadow:0 0 16px var(--c); animation:fill 1.5s var(--ease) .35s both;
}
@keyframes fill{ from{ width:0; } }
.pv-scale{ display:flex; justify-content:space-between; color:var(--muted); font-size:.78rem; margin-top:8px; }
.pv-rec{ margin:10px 0 0; line-height:1.55; color:var(--ink); }

/* ── Summary banner ───────────────────────────────────── */
.pv-summary{
  display:flex; align-items:center; gap:16px; padding:16px 24px; margin-bottom:1.2rem;
}
.pv-summary-icon{
  width:44px; height:44px; flex:none; display:grid; place-items:center; border-radius:14px;
  font-size:1.3rem;
}
.pv-summary-text{ font-weight:600; color:var(--ink); line-height:1.4; }
.pv-summary-text small{ display:block; font-weight:400; color:var(--muted); font-size:.88rem; margin-top:2px; }

/* ── Buttons ──────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button, [data-testid="stFileUploader"] button{
  position:relative; overflow:hidden; border-radius:16px; padding:.7rem 1.4rem;
  font:600 1rem 'Instrument Sans',sans-serif; color:var(--ink);
  background:linear-gradient(140deg, rgba(255,255,255,.92), rgba(255,232,242,.75));
  border:1px solid var(--stroke); backdrop-filter:blur(16px);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.95), 0 10px 28px -14px rgba(219,112,155,.35);
  transition:transform .35s var(--ease), box-shadow .35s var(--ease), border-color .35s var(--ease), filter .35s;
}
.stButton > button p, .stDownloadButton > button p{ color:inherit; margin:0; }
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFileUploader"] button:hover{
  transform:translateY(-3px); border-color:rgba(244,114,182,.6); color:var(--ink);
  box-shadow:inset 0 1px 0 #fff, 0 18px 36px -14px rgba(244,114,182,.45);
}
.stButton > button:active, .stDownloadButton > button:active{ transform:translateY(0) scale(.97); transition-duration:.12s; }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible{ outline:2px solid var(--beam); outline-offset:3px; }
.stButton > button::after{
  content:""; position:absolute; inset:0; pointer-events:none;
  background:linear-gradient(110deg, transparent 35%, rgba(255,255,255,.75) 50%, transparent 65%);
  transform:translateX(-120%); transition:transform .8s var(--ease);
}
.stButton > button:hover::after{ transform:translateX(120%); }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"]{
  color:#fff; font-size:1.05rem; padding:.85rem 1.4rem; border-color:rgba(255,255,255,.65);
  background:linear-gradient(135deg, #FF85B3, #F472B6 55%, #EC4899);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.7), 0 16px 38px -12px rgba(244,114,182,.65);
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover{
  color:#fff; filter:brightness(1.05); box-shadow:inset 0 1px 0 rgba(255,255,255,.85), 0 22px 46px -12px rgba(244,114,182,.8);
}

/* ── Slider ───────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"]{
  width:22px; height:22px; border:1px solid rgba(244,114,182,.85);
  background:radial-gradient(circle at 30% 28%, #fff, #FFD6E8 55%, #F472B6);
  box-shadow:0 0 0 6px rgba(244,114,182,.18), 0 6px 16px rgba(180,90,130,.25);
  transition:transform .25s var(--ease), box-shadow .25s var(--ease);
}
[data-testid="stSlider"] [role="slider"]:hover, [data-testid="stSlider"] [role="slider"]:focus-visible{
  transform:scale(1.18); box-shadow:0 0 0 10px rgba(244,114,182,.24), 0 8px 20px rgba(180,90,130,.35);
}
[data-testid="stSliderThumbValue"]{ font-family:'Bricolage Grotesque',sans-serif; font-weight:700; color:var(--signal); }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"]{ color:var(--muted); }

/* ── Inputs ───────────────────────────────────────────── */
[data-baseweb="select"] > div, [data-baseweb="input"]{
  background:rgba(255,255,255,.82) !important; border:1px solid var(--stroke) !important;
  border-radius:14px !important; backdrop-filter:blur(12px); color:var(--ink) !important;
  transition:border-color .3s var(--ease), box-shadow .3s var(--ease);
}
[data-baseweb="select"] > div:hover, [data-baseweb="input"]:focus-within{
  border-color:rgba(244,114,182,.75) !important; box-shadow:0 0 0 4px rgba(244,114,182,.16);
}
[data-baseweb="input"] input{ background:transparent !important; color:var(--ink) !important; }
div[data-baseweb="popover"] > div{
  background:rgba(255,248,251,.95) !important; backdrop-filter:blur(26px) saturate(170%);
  border:1px solid var(--stroke); border-radius:16px; box-shadow:0 24px 50px -16px rgba(219,112,155,.3);
}

/* ── Uploader ─────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"]{
  background:linear-gradient(140deg, rgba(255,255,255,.88), rgba(255,235,243,.6));
  border:1.5px dashed rgba(244,114,182,.45); border-radius:22px; padding:1.6rem;
  backdrop-filter:blur(20px); color:var(--ink);
  transition:border-color .4s var(--ease), transform .4s var(--ease), box-shadow .4s var(--ease);
}
[data-testid="stFileUploaderDropzone"]:hover{
  border-color:var(--beam); transform:translateY(-2px); box-shadow:0 20px 44px -22px rgba(244,114,182,.45);
}

/* ── Tabs ─────────────────────────────────────────────── */
[data-baseweb="tab-list"]{
  gap:6px; padding:6px; width:fit-content; max-width:100%; border-radius:18px;
  background:rgba(255,238,245,.75); border:1px solid var(--stroke); backdrop-filter:blur(18px);
}
button[data-baseweb="tab"]{
  height:auto; padding:.55rem 1.3rem; border-radius:13px; background:transparent; color:var(--muted);
  font:600 .98rem 'Bricolage Grotesque',sans-serif; transition:all .4s var(--ease);
}
button[data-baseweb="tab"]:hover{ color:var(--ink); background:rgba(255,255,255,.6); }
button[data-baseweb="tab"][aria-selected="true"]{
  color:var(--ink); background:linear-gradient(140deg, #FFFFFF, #FFE4EF);
  box-shadow:inset 0 1px 0 #fff, 0 8px 20px -8px rgba(244,114,182,.45);
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"]{ display:none; }
[data-baseweb="tab-panel"]{ padding-top:1.2rem; animation:rise .55s var(--ease) both; }

/* ── Images, alerts, misc ─────────────────────────────── */
[data-testid="stImage"] img{
  border-radius:22px; border:1px solid var(--stroke);
  box-shadow:0 34px 60px -28px rgba(180,90,130,.3), inset 0 1px 0 rgba(255,255,255,.9);
}
[data-testid="stAlert"]{
  border-radius:18px; background:rgba(255,255,255,.85); border:1px solid var(--stroke); backdrop-filter:blur(18px); color:var(--ink);
}
[data-testid="stSpinner"]{ padding:12px 18px; border-radius:16px; background:rgba(255,255,255,.85); border:1px solid var(--stroke); color:var(--ink); }
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-thumb{ background:rgba(244,114,182,.35); border-radius:10px; }

/* ── Divider ──────────────────────────────────────────── */
.pv-divider{
  border:none; height:1px; margin:1.6rem 0;
  background:linear-gradient(90deg, transparent, var(--stroke), transparent);
}

@media (prefers-reduced-motion:reduce){
  *, *::before, *::after{ animation-duration:.01ms !important; animation-delay:0s !important; transition-duration:.01ms !important; }
}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def inject_css():
    css = "\n".join(line for line in CSS.splitlines() if line.strip())
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def html(fragment: str):
    """Render an HTML fragment."""
    flat = " ".join(line.strip() for line in fragment.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


def esc(value) -> str:
    return _html.escape(str(value))


def get_boxes(analysis, *names):
    """Return a list of [x1, y1, x2, y2] boxes, or None if the engine doesn't expose them."""
    for name in names:
        if hasattr(analysis, name):
            raw = getattr(analysis, name)
            if raw is None:
                return []
            if hasattr(raw, "cpu"):
                raw = raw.cpu().numpy()
            try:
                arr = np.asarray(raw, dtype=float)
                if arr.size == 0:
                    return []
                return arr.reshape(-1, 4).tolist()
            except (TypeError, ValueError):
                return []
    return None


def render_view(img_rgb, vacant=None, occupied=None):
    """Spotlight view: the lot is dimmed and only the requested slots stay lit and outlined."""
    base = Image.fromarray(img_rgb).convert("RGB")
    w, h = base.size
    dimmed = ImageEnhance.Brightness(ImageEnhance.Color(base).enhance(0.45)).enhance(0.32)

    mask = Image.new("L", base.size, 0)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_draw, over_draw = ImageDraw.Draw(mask), ImageDraw.Draw(overlay)
    line = max(2, round(min(w, h) / 320))

    for boxes, color in ((vacant, VACANT_RGB), (occupied, OCCUPIED_RGB)):
        for x1, y1, x2, y2 in boxes or []:
            ix1, ix2 = sorted((max(0, int(round(x1))), min(w, int(round(x2)))))
            iy1, iy2 = sorted((max(0, int(round(y1))), min(h, int(round(y2)))))
            bw, bh = ix2 - ix1, iy2 - iy1
            if bw < 2 or bh < 2:
                continue
            rect = (ix1, iy1, ix2, iy2)
            radius = max(1, min(line * 2, bw // 2, bh // 2))
            mask_draw.rounded_rectangle(rect, radius=radius, fill=255)
            over_draw.rounded_rectangle(
                rect,
                radius=radius,
                fill=color + (58,),
                outline=color + (255,),
                width=min(line, bw // 2, bh // 2),
            )

    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    lit = Image.composite(base, dimmed, mask).convert("RGBA")
    return Image.alpha_composite(lit, overlay).convert("RGB")


def to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@st.cache_resource
def load_engine():
    return ParkVisionEngine(
        custom_weights_path=r"runs/detect/train-11/weights/best.pt",
        general_weights_path="yolov8s.pt",
    )


# ──────────────────────────────────────────────────────────────────────────────
# SVG Icons (inline, no emoji dependencies)
# ──────────────────────────────────────────────────────────────────────────────
ICON_CAR = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 17h14M5 17a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h1l2-3h8l2 3h1a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2M5 17v2m14-2v2"/><circle cx="7.5" cy="14.5" r="1.5"/><circle cx="16.5" cy="14.5" r="1.5"/></svg>'

ICON_UPLOAD = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>'

ICON_SCAN = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="3"/></svg>'

ICON_CHART = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>'


# ──────────────────────────────────────────────────────────────────────────────
# UI blocks
# ──────────────────────────────────────────────────────────────────────────────
def hero():
    html(f"""
    <div class="glass pv-hero">
      <div class="pv-mark">{ICON_CAR}</div>
      <div>
        <div class="pv-title">ParkVision AI</div>
        <div class="pv-sub">Detect vacant &amp; occupied parking slots from a single photo — powered by YOLOv8.</div>
      </div>
    </div>
    """)


def empty_state():
    html(f"""
    <div class="pv-steps">
      <div class="glass pv-step">
        <div class="pv-step-num">1</div>
        <span>Upload a photo</span>
        <small>JPG, PNG or WebP of any parking lot — aerial or angled.</small>
      </div>
      <div class="glass pv-step">
        <div class="pv-step-num">2</div>
        <span>Press Detect slots</span>
        <small>Tweak engine &amp; confidence in the sidebar first if needed.</small>
      </div>
      <div class="glass pv-step">
        <div class="pv-step-num">3</div>
        <span>Explore results</span>
        <small>Toggle between vacant, occupied, or combined slot views.</small>
      </div>
    </div>
    """)


def file_panel(name, size, mode, conf, cap, stale):
    cap_text = "Auto capacity" if cap == 0 else f"Capacity {cap}"
    warn = '<span class="pv-chip warn">Settings changed — detect again to update</span>' if stale else ""
    html(f"""
    <div class="glass pv-panel">
      <div class="pv-h">{esc(name)}</div>
      <div class="pv-chips">
        <span class="pv-chip">{size[0]} × {size[1]} px</span>
        <span class="pv-chip">{esc(mode)}</span>
        <span class="pv-chip">Confidence {conf:.0%}</span>
        <span class="pv-chip">{cap_text}</span>
        {warn}
      </div>
    </div>
    """)


def summary_banner(analysis):
    """A quick one-line status banner shown above the metrics."""
    rate = max(0.0, min(100.0, float(analysis.occupancy_rate)))
    hex_c = STATUS_HEX.get(str(analysis.status_color).lower(), "#F472B6")

    if rate < 40:
        icon, bg = "🟢", "rgba(46,196,134,.12)"
        msg = f"Plenty of space — {int(analysis.available_slots)} slots open"
    elif rate < 75:
        icon, bg = "🟡", "rgba(251,191,36,.12)"
        msg = f"Filling up — {int(analysis.available_slots)} slots remaining"
    else:
        icon, bg = "🔴", "rgba(232,74,127,.12)"
        msg = f"Nearly full — only {int(analysis.available_slots)} slots left"

    html(f"""
    <div class="glass pv-summary reveal" style="--i:0">
      <div class="pv-summary-icon" style="background:{bg}; font-size:1.5rem;">{icon}</div>
      <div class="pv-summary-text">
        {esc(msg)}
        <small>{esc(analysis.congestion_level)} · {rate:.1f}% occupied · {esc(analysis.engine_mode)}</small>
      </div>
    </div>
    """)


def metrics_block(a):
    rate = max(0.0, min(100.0, float(a.occupancy_rate)))
    whole, dec = f"{rate:.1f}".split(".")
    hex_c = STATUS_HEX.get(str(a.status_color).lower(), "#F472B6")

    html(f"""
    <div class="pv-grid">
      <div class="glass pv-metric reveal" style="--i:0;--accent:var(--beam)">
        <div class="pv-label">Total slots</div>
        <div class="pv-value"><span class="count" style="--i:0;--target:{int(a.total_slots)}"></span></div>
      </div>
      <div class="glass pv-metric reveal" style="--i:1;--accent:var(--signal)">
        <div class="pv-label">Occupied</div>
        <div class="pv-value"><span class="count" style="--i:1;--target:{int(a.occupied_slots)}"></span></div>
      </div>
      <div class="glass pv-metric reveal" style="--i:2;--accent:var(--lagoon)">
        <div class="pv-label">Available</div>
        <div class="pv-value"><span class="count" style="--i:2;--target:{int(a.available_slots)}"></span></div>
      </div>
      <div class="glass pv-metric reveal" style="--i:3;--accent:var(--dusk)">
        <div class="pv-label">Occupancy</div>
        <div class="pv-value"><span class="count" style="--i:3;--target:{whole}"></span><small>.{dec}%</small></div>
      </div>
    </div>
    <div class="pv-split">
      <div class="glass pv-panel reveal" style="--i:4">
        <div class="pv-row">
          <span class="pv-h">Congestion</span>
          <span class="pv-chip"><i class="pv-dot" style="--c:{hex_c}"></i>{esc(a.congestion_level)}</span>
        </div>
        <div class="pv-bar" style="--w:{rate:.1f}%;--c:{hex_c}"><span></span></div>
        <div class="pv-scale"><span>0%</span><span>100%</span></div>
      </div>
      <div class="glass pv-panel reveal" style="--i:5">
        <span class="pv-h">Recommendation</span>
        <p class="pv-rec">{esc(a.recommendation)}</p>
      </div>
    </div>
    """)


def view_tab(key, analysis, img_rgb, vacant, occupied, want_vacant, want_occupied, label, cached_views=None):
    needed = ([vacant] if want_vacant else []) + ([occupied] if want_occupied else [])

    if any(b is None for b in needed):
        if want_vacant and want_occupied:
            st.image(analysis.annotated_image, channels="BGR", use_container_width=True)
            st.caption("Showing annotated image. Return vacant_boxes and occupied_boxes from engine for filtered views.")
        else:
            msg = f"The {label.lower()} view needs slot boxes from the engine."
            st.info(msg)
        return

    if cached_views and key in cached_views:
        view, png_bytes = cached_views[key]
    else:
        view = render_view(
            img_rgb,
            vacant=vacant if want_vacant else None,
            occupied=occupied if want_occupied else None,
        )
        png_bytes = to_png(view)

    st.image(view, use_container_width=True)

    parts = []
    if want_vacant:
        parts.append(f"{len(vacant)} vacant")
    if want_occupied:
        parts.append(f"{len(occupied)} occupied")
    shown = sum(len(b) for b in needed)
    c1, c2 = st.columns([3, 1])
    with c1:
        caption_txt = " and ".join(parts) + " slots highlighted" if shown else "No slot outlines to show in this view."
        st.caption(caption_txt)
    with c2:
        st.download_button(
            "⬇ Save image",
            png_bytes,
            file_name=f"parkvision_{key}.png",
            mime="image/png",
            key=f"dl_{key}",
            use_container_width=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────
inject_css()
engine = load_engine()

with st.sidebar:
    st.markdown('<div class="pv-side-title">Detection settings</div>', unsafe_allow_html=True)
    inference_mode = st.selectbox(
        "Detection engine",
        ["Auto (Hybrid)", "Universal Generalization Mode", "Custom Dataset Mode"],
        help="Auto uses your custom PKLot weights first and switches to the Universal Engine for unseen images.",
    )
    conf_slider = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.80,
        value=0.20,
        step=0.05,
        help="Lower values find more cars but can add false detections.",
    )
    manual_cap = st.number_input(
        "Total lot capacity (0 = auto-detect)",
        min_value=0,
        max_value=500,
        value=0,
        help="Set total slots manually for unmapped Google images if needed.",
    )
    html("""
    <div class="glass pv-tip">Changing a setting doesn't re-run detection. Press <b>Detect slots</b> to apply it.</div>
    """)
    html("""
    <div class="pv-badge">✦ ParkVision v2.0</div>
    """)

hero()

uploaded_file = st.file_uploader("Upload a parking lot image", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is None:
    st.session_state.pop("result", None)
    empty_state()
    st.stop()

file_key = f"{uploaded_file.name}:{uploaded_file.size}"
settings = (inference_mode, conf_slider, manual_cap)

if st.session_state.get("cached_file_key") != file_key:
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)
    st.session_state.cached_file_key = file_key
    st.session_state.cached_image = image
    st.session_state.cached_img_rgb = img_rgb
else:
    image = st.session_state.cached_image
    img_rgb = st.session_state.cached_img_rgb

result = st.session_state.get("result")
if result and result["file_key"] != file_key:
    st.session_state.pop("result")
    result = None
stale = bool(result) and result["settings"] != settings

if result is None:
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.image(image, use_container_width=True)
    with right:
        file_panel(uploaded_file.name, image.size, inference_mode, conf_slider, manual_cap, stale)
        st.write("")
        detect_clicked = st.button("Detect slots", type="primary", use_container_width=True, key="detect_first")
else:
    left, right = st.columns([3, 1], gap="large")
    with left:
        file_panel(uploaded_file.name, image.size, inference_mode, conf_slider, manual_cap, stale)
    with right:
        detect_clicked = st.button("Detect again", type="primary", use_container_width=True, key="detect_again")

if detect_clicked:
    with st.spinner("Analyzing parking lot layout…"):
        analysis = engine.analyze(
            image_rgb=img_rgb,
            conf_threshold=conf_slider,
            mode=inference_mode,
            manual_capacity=manual_cap,
        )
        v_boxes = get_boxes(analysis, "vacant_boxes", "empty_boxes", "available_boxes", "free_boxes")
        o_boxes = get_boxes(analysis, "occupied_boxes")
        cached_views = {}
        if v_boxes is not None and o_boxes is not None:
            for k, wv, wo in (("vacant", True, False), ("occupied", False, True), ("both", True, True)):
                v_img = render_view(img_rgb, vacant=v_boxes if wv else None, occupied=o_boxes if wo else None)
                cached_views[k] = (v_img, to_png(v_img))

    st.session_state.result = {
        "file_key": file_key,
        "settings": settings,
        "analysis": analysis,
        "views": cached_views,
    }
    st.session_state.just_detected = True
    st.rerun()

if st.session_state.pop("just_detected", False):
    st.toast("Detection complete", icon="✅")

if result is None:
    st.stop()

analysis = result["analysis"]
cached_views = result.get("views", {})

# ── Summary banner + metrics ──────────────────────────────────────────────────
summary_banner(analysis)
html('<hr class="pv-divider">')
metrics_block(analysis)

# ── View tabs ─────────────────────────────────────────────────────────────────
vacant_boxes = get_boxes(analysis, "vacant_boxes", "empty_boxes", "available_boxes", "free_boxes")
occupied_boxes = get_boxes(analysis, "occupied_boxes")

tab_vacant, tab_occupied, tab_both = st.tabs(["🟢 Vacant", "🔴 Occupied", "🔵 Both"])
with tab_vacant:
    view_tab("vacant", analysis, img_rgb, vacant_boxes, occupied_boxes, True, False, "Vacant", cached_views)
with tab_occupied:
    view_tab("occupied", analysis, img_rgb, vacant_boxes, occupied_boxes, False, True, "Occupied", cached_views)
with tab_both:
    view_tab("both", analysis, img_rgb, vacant_boxes, occupied_boxes, True, True, "Both", cached_views)
