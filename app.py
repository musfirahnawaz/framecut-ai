import html
import math
import os
import tempfile
import time

import cv2
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from moviepy import (
    CompositeVideoClip,
    TextClip,
    VideoFileClip,
    concatenate_videoclips,
)
from pydantic import BaseModel, Field

load_dotenv()

MODELS = ["gemma-4-26b-a4b-it", "gemma-4-31b-it"]
FONT = "C:/Windows/Fonts/arial.ttf"

# ---------------------------------------------------------------- styling
CSS = """
.stApp {
  background:
    radial-gradient(1000px 520px at 8% -10%, #3b1a7a 0%, transparent 60%),
    radial-gradient(800px 460px at 100% 0%, #0b4a6e 0%, transparent 55%),
    #0F0B1E;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 880px; }
section[data-testid="stSidebar"] {
  background: rgba(26, 20, 51, 0.92);
  border-right: 1px solid rgba(255,255,255,0.08);
}
.hero { display: flex; align-items: center; gap: 18px; margin-bottom: 14px; }
.logo {
  width: 68px; height: 68px; border-radius: 20px; font-size: 34px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, #8B5CF6, #EC4899 55%, #F59E0B);
  box-shadow: 0 10px 30px rgba(139, 92, 246, 0.5);
}
.brand {
  font-size: 2.6rem; font-weight: 800; line-height: 1.05;
  background: linear-gradient(90deg, #ffffff, #c4b5fd 55%, #f9a8d4);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.tagline { color: #b8aee0; margin-top: 6px; font-size: 1rem; }
.steps { display: flex; gap: 10px; flex-wrap: wrap; margin: 6px 0 22px 0; }
.step {
  padding: 8px 14px; border-radius: 999px; font-size: 0.9rem; color: #e9e3ff;
  background: rgba(139, 92, 246, 0.14); border: 1px solid rgba(139, 92, 246, 0.35);
}
.step b {
  display: inline-block; width: 22px; height: 22px; line-height: 22px;
  text-align: center; border-radius: 50%; margin-right: 8px; color: #fff;
  background: linear-gradient(135deg, #8B5CF6, #EC4899);
}
.section-title { font-size: 1.25rem; font-weight: 700; margin: 26px 0 8px 0; color: #ffffff; }
.card {
  background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.10);
  border-radius: 18px; padding: 18px 20px; margin: 12px 0;
}
.clip-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
.clip-num {
  font-weight: 800; font-size: 1.05rem; padding: 4px 12px; border-radius: 10px; color: #fff;
  background: linear-gradient(135deg, #8B5CF6, #EC4899);
}
.pill {
  font-size: 0.82rem; padding: 4px 10px; border-radius: 999px; color: #d9d0ff;
  background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.12);
}
.summary { color: #e6e0ff; margin: 6px 0 12px 0; line-height: 1.5; }
.caption-bubble {
  border-left: 4px solid #EC4899; background: rgba(236, 72, 153, 0.12);
  padding: 10px 14px; border-radius: 10px; color: #ffe4f1; font-weight: 600;
}
.stats { display: flex; gap: 12px; flex-wrap: wrap; margin: 10px 0; }
.stat {
  flex: 1; min-width: 140px; padding: 14px 16px; border-radius: 16px; text-align: center;
  background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.10);
}
.stat .num { font-size: 1.6rem; font-weight: 800; color: #ffffff; }
.stat .lbl { font-size: 0.8rem; color: #b8aee0; }
.stButton > button, .stDownloadButton > button {
  background: linear-gradient(135deg, #8B5CF6, #EC4899); color: #ffffff; border: 0;
  border-radius: 14px; padding: 0.75rem 1.5rem; font-weight: 700; font-size: 1rem;
  box-shadow: 0 8px 24px rgba(139, 92, 246, 0.4);
  transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  color: #ffffff; border: 0; transform: translateY(-2px); filter: brightness(1.1);
  box-shadow: 0 12px 30px rgba(236, 72, 153, 0.45);
}
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(0); color: #ffffff; }
.stButton > button:focus, .stDownloadButton > button:focus { color: #ffffff; border: 0; }
[data-testid="stFileUploaderDropzone"] {
  background: rgba(139, 92, 246, 0.08); border: 2px dashed rgba(139, 92, 246, 0.6);
  border-radius: 18px;
}
[data-testid="stFileUploaderDropzone"]:hover { background: rgba(139, 92, 246, 0.14); }
[data-testid="stVideo"], video { border-radius: 16px; }
[data-testid="stExpander"] {
  border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 14px;
  background: rgba(255, 255, 255, 0.03);
}
.footer { text-align: center; color: #8d83bd; font-size: 0.85rem; margin: 40px 0 10px 0; }
"""

STEPS = (
    '<div class="steps">'
    '<div class="step"><b>1</b>Upload video</div>'
    '<div class="step"><b>2</b>Gemma picks highlights</div>'
    '<div class="step"><b>3</b>Render reel with captions</div>'
    "</div>"
)


def clip_card(i, c):
    return (
        '<div class="card"><div class="clip-head">'
        f'<span class="clip-num">Clip {i}</span>'
        f'<span class="pill">⏱ {c.start_time:.1f}s → {c.end_time:.1f}s</span>'
        f'<span class="pill">{c.end_time - c.start_time:.1f}s long</span></div>'
        f'<div class="summary">{html.escape(c.summary)}</div>'
        f'<div class="caption-bubble">💬 {html.escape(c.caption)}</div></div>'
    )


def stat_tiles(frames_n, duration, clips_n):
    return (
        '<div class="stats">'
        f'<div class="stat"><div class="num">{frames_n}</div><div class="lbl">Frames sampled</div></div>'
        f'<div class="stat"><div class="num">{duration:.0f}s</div><div class="lbl">Video length</div></div>'
        f'<div class="stat"><div class="num">{clips_n}</div><div class="lbl">Highlights found</div></div>'
        "</div>"
    )


# ------------------------------------------------------------------ logic
class Clip(BaseModel):
    start_time: float = Field(description="Clip start in seconds")
    end_time: float = Field(description="Clip end in seconds")
    summary: str = Field(description="Key visual actions in this clip")
    caption: str = Field(description="Short overlay caption, max 10 words")


class Highlights(BaseModel):
    clips: list[Clip]


def extract_frames(path, max_frames, width=512):
    """Sample about 1 frame per second (fewer if the video is long)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if total <= 0:  # some WebM files do not report a frame count
        total = 0
        while cap.grab():
            total += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    duration = total / fps
    step = max(1, math.ceil(duration / max_frames))

    frames = []
    for sec in range(0, int(duration) + 1, step):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (width, int(h * width / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            frames.append((sec, buf.tobytes()))
    cap.release()
    return frames, duration


def _generate(client, model, contents, config=None):
    """Call Gemma, retrying once on temporary server errors."""
    last = None
    for attempt in range(2):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as e:
            last = e
            time.sleep(3)
    raise last


def ask_gemma(frames, duration):
    client = genai.Client()
    prompt = (
        f"You are a video editor. The video is {duration:.0f} seconds long. "
        "Below are frames sampled from it, each labelled with its time in seconds. "
        "Pick the TOP 2 most engaging, shareable moments. "
        "Each clip must be 3 to 10 seconds long, inside the video, and the two "
        "clips must not overlap. For each clip give start_time and end_time "
        "(in seconds), a summary of the key visual actions, and a short overlay "
        "caption (max 10 words)."
    )
    contents = [prompt]
    for sec, data in frames:
        contents.append(f"Frame at {sec}s:")
        contents.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))

    plain_prompt = prompt + (
        ' Reply with JSON only, like: {"clips":[{"start_time":0,'
        '"end_time":5,"summary":"...","caption":"..."}]}'
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Highlights,
    )

    last_error = None
    for model in MODELS:
        try:  # 1) structured output
            r = _generate(client, model, contents, config)
            return Highlights.model_validate_json(r.text)
        except Exception as e:
            last_error = e
        try:  # 2) plain JSON, parsed by us
            r = _generate(client, model, [plain_prompt] + contents[1:])
            text = r.text.strip()
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return Highlights.model_validate_json(text)
        except Exception as e:
            last_error = e
    raise last_error


def clean_clips(clips, duration):
    """Keep timestamps inside the video and make sure end > start."""
    out = []
    for c in clips:
        start = max(0.0, min(c.start_time, duration - 1))
        end = max(start + 1, min(c.end_time, duration))
        out.append(Clip(start_time=start, end_time=end,
                        summary=c.summary, caption=c.caption))
    return out[:2]


def cut(video, start, end):
    # MoviePy 2.x uses subclipped, 1.x uses subclip
    if hasattr(video, "subclipped"):
        return video.subclipped(start, end)
    return video.subclip(start, end)


def add_caption(clip, text):
    txt = TextClip(
        font=FONT,
        text=text,
        font_size=max(24, int(clip.h * 0.05)),
        color="white",
        stroke_color="black",
        stroke_width=2,
        method="caption",
        size=(int(clip.w * 0.9), None),
        margin=(20, 20),
    )
    txt = txt.with_position(("center", 0.78), relative=True).with_duration(clip.duration)
    return CompositeVideoClip([clip, txt])


def render_clips(video_path, clips):
    out_dir = tempfile.mkdtemp()
    source = VideoFileClip(video_path)
    if source.h > 720:  # big files render much faster at 720p
        source = (source.resized(height=720) if hasattr(source, "resized")
                  else source.resize(height=720))
    parts, files, warning = [], [], None
    for i, c in enumerate(clips, 1):
        piece = cut(source, c.start_time, min(c.end_time, source.duration))
        try:
            piece = add_caption(piece, c.caption)
        except Exception as e:
            warning = f"Caption burn-in failed, using plain trim: {e}"
        path = os.path.join(out_dir, f"clip_{i}.mp4")
        piece.write_videofile(path, codec="libx264", audio_codec="aac", logger=None)
        parts.append(piece)
        files.append(path)

    reel_path = os.path.join(out_dir, "highlight_reel.mp4")
    concatenate_videoclips(parts).write_videofile(
        reel_path, codec="libx264", audio_codec="aac", logger=None
    )
    source.close()
    return files, reel_path, warning


# --------------------------------------------------------------------- UI
st.set_page_config(page_title="FrameCut AI", page_icon="🎬", layout="centered")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><div class="logo">🎬</div><div>'
    '<div class="brand">FrameCut AI</div>'
    '<div class="tagline">Turn raw footage into viral highlights with burned-in '
    "captions, powered by Gemma 4.</div></div></div>",
    unsafe_allow_html=True,
)
st.markdown(STEPS, unsafe_allow_html=True)

st.sidebar.markdown("### ⚙️ Settings")
max_frames = st.sidebar.slider(
    "Frames sent to Gemma", 5, 60, 20, help="Fewer frames = faster response"
)
st.sidebar.markdown("---")
st.sidebar.caption("Model: gemma-4-26b-a4b-it (open-weight, Apache 2.0)")

st.markdown('<div class="section-title">📤 Upload your video</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "MP4 or WebM, up to 60 seconds",
    type=["mp4", "webm"],
    label_visibility="collapsed",
)

if uploaded:
    st.video(uploaded)
    if st.button("✨ Generate Highlight"):
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            video_path = tmp.name

        with st.spinner("Extracting frames..."):
            frames, duration = extract_frames(video_path, max_frames)

        try:
            with st.spinner("Gemma is picking the highlights..."):
                result = ask_gemma(frames, duration)
            st.session_state["video_path"] = video_path
            st.session_state["clips"] = clean_clips(result.clips, duration)
            st.session_state["meta"] = (len(frames), duration)
            st.session_state.pop("rendered", None)
        except Exception as e:
            st.error(f"Gemma call failed: {e}")

if "clips" in st.session_state:
    clips = st.session_state["clips"]
    frames_n, duration = st.session_state.get("meta", (0, 0))

    st.markdown('<div class="section-title">🧠 Gemma\'s picks</div>', unsafe_allow_html=True)
    st.markdown(stat_tiles(frames_n, duration, len(clips)), unsafe_allow_html=True)
    for i, c in enumerate(clips, 1):
        st.markdown(clip_card(i, c), unsafe_allow_html=True)
    with st.expander("View raw JSON"):
        st.json([c.model_dump() for c in clips])

    st.markdown('<div class="section-title">🎞️ Render</div>', unsafe_allow_html=True)
    if st.button("🎞️ Render highlight reel"):
        with st.spinner("Trimming clips and burning in captions..."):
            try:
                st.session_state["rendered"] = render_clips(
                    st.session_state["video_path"], clips
                )
            except Exception as e:
                st.error(f"Rendering failed: {e}")

if "rendered" in st.session_state:
    files, reel, warning = st.session_state["rendered"]
    if warning:
        st.warning(warning)
    st.markdown('<div class="section-title">🏆 Your highlight reel</div>', unsafe_allow_html=True)
    with open(reel, "rb") as f:
        reel_bytes = f.read()
    st.video(reel_bytes)
    st.download_button(
        "⬇️ Download reel", reel_bytes, file_name="highlight_reel.mp4", mime="video/mp4"
    )
    with st.expander("Individual clips"):
        for i, path in enumerate(files, 1):
            st.write(f"Clip {i}")
            st.video(path)

st.markdown(
    '<div class="footer">Built with Gemma 4 · Streamlit · OpenCV · MoviePy</div>',
    unsafe_allow_html=True,
)