# Transcript

A free, open-source macOS audio transcription app with speaker diarization. A simpler alternative to MacWhisper, built for Apple Silicon.

## Features

- Local audio transcription using [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (Whisper large-v3-turbo on MLX)
- Speaker diarization using [pyannote-audio](https://github.com/pyannote/pyannote-audio)
- Runs entirely on-device — no cloud APIs, no data leaves your Mac
- Optimized for Apple Silicon (M1–M4)

## Requirements

- macOS with Apple Silicon (M1 or later)
- Python 3.10+
- A free [HuggingFace account](https://huggingface.co/) (for pyannote model access)

## Setup

```bash
# Clone the repo
git clone https://github.com/lorenzoorsingher/transcript.git
cd transcript

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### HuggingFace Token (for speaker diarization)

1. Create a free account at [huggingface.co](https://huggingface.co/)
2. Accept the terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Set it as an environment variable:
   ```bash
   export HF_TOKEN=your_token_here
   ```

## Usage

```bash
python transcribe.py path/to/audio.wav
```

## Tech Stack

| Component | Library | Why |
|-----------|---------|-----|
| Transcription | mlx-whisper (large-v3-turbo) | Fastest Whisper on Apple Silicon (~2x faster than whisper.cpp) |
| Diarization | pyannote-audio | Best open-source speaker diarization |
| Framework | MLX | Apple's ML framework, native Metal acceleration |

## License

MIT
