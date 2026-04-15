# Transcript

A free, open-source macOS audio transcription app with speaker diarization. A simpler alternative to MacWhisper, built for Apple Silicon.

## Features

- Local audio transcription using [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (Whisper large-v3-turbo on MLX)
- Speaker diarization using [pyannote-audio](https://github.com/pyannote/pyannote-audio)
- Runs entirely on-device — no cloud APIs, no data leaves your Mac
- Optimized for Apple Silicon (M1–M5)

## Requirements

- macOS with Apple Silicon (M1 or later, including M5)
- Python 3.11+
- A free [HuggingFace account](https://huggingface.co/) (for pyannote model access)

## Setup

```bash
# Clone the repo
git clone https://github.com/LoriTira/transcript.git
cd transcript

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### HuggingFace Token (for speaker diarization)

1. Create a free account at [huggingface.co](https://huggingface.co/)
2. Accept the terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Read access is sufficient)
4. Create a `.env` file in the project root:
   ```bash
   echo "HF_TOKEN=your_token_here" > .env
   ```

## Usage

```bash
# Launch the web UI
python app.py
```

This opens a Gradio web interface where you can:
- Upload audio files or record from your microphone
- Optionally specify the number of speakers
- View the diarized transcript with speaker labels and timestamps

## Tech Stack

| Component | Library | Why |
|-----------|---------|-----|
| Transcription | mlx-whisper (large-v3-turbo) | Fastest Whisper on Apple Silicon via MLX, native Metal acceleration |
| Diarization | pyannote-audio 3.1 | Best open-source speaker diarization, MPS-accelerated on Apple Silicon |
| Deep Learning | PyTorch 2.5+ with MPS backend | Full Metal Performance Shaders support for M-series GPUs |
| Framework | MLX | Apple's ML framework, optimized for unified memory on M1–M5 |
| UI | Gradio 5 | Modern web interface with file upload and mic recording |

## License

MIT
