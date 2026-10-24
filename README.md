# 🎬 FrameCut AI

**FrameCut AI helps content creators turn a short raw video into highlight clips with burned-in captions, using Gemma 4.**

Creators spend hours scrubbing through raw footage to find the best moments. FrameCut AI does it in one click: upload a clip, and Gemma 4 picks the top 2 moments, writes short captions, and the app renders a highlight reel.

## How it works
1. **Upload** an MP4/WebM video (up to 60 seconds).
2. **Frames are sampled** with OpenCV (about 1 frame per second, resized to keep requests light).
3. **Gemma 4 looks at the frames** and returns structured JSON (validated with a Pydantic schema): `start_time`, `end_time`, a summary of the key visual actions, and an overlay caption for each of the top 2 clips.
4. **MoviePy trims** the clips and burns the captions into the video, then joins them into a highlight reel you can download.

## Model
- **Gemma 4** (`gemma-4-26b-a4b-it`, with `gemma-4-31b-it` as a fallback), an open-weight model, called through the Gemini API.
- License: Apache 2.0, see [GEMMA_LICENSE_LINK]
- Gemma 4 does the core work: it understands the video frames, decides which moments are the highlights, and writes the captions.

## Run it yourself
```bash
git clone https://github.com/YOUR_USERNAME/framecut-ai.git
cd framecut-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Create a `.env` file with your Gemini API key (from Google AI Studio):
```
GEMINI_API_KEY=your_key_here
```
Then start the app:
```bash
streamlit run app.py
```

## Tech stack
Python, Streamlit, OpenCV, MoviePy, Pydantic, google-genai SDK.

## License
MIT, see `LICENSE`.
