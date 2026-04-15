"""
Core transcription + diarization pipeline.

Architecture:
  Audio → mlx-whisper (transcription with timestamps)
        → pyannote (speaker diarization)
        → merge: assign speaker labels to transcript segments
"""

import os
import tempfile
from pathlib import Path

import mlx_whisper
import torch
import torchaudio
from dotenv import load_dotenv
from pyannote.audio import Pipeline as DiarizationPipeline

load_dotenv()

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


def get_device():
    """Return best available torch device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Lazy-loaded singletons so models are only downloaded/loaded once
# ---------------------------------------------------------------------------
_diarization_pipeline = None


def _get_diarization_pipeline():
    global _diarization_pipeline
    if _diarization_pipeline is None:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN not set. Export your HuggingFace token as HF_TOKEN."
            )
        _diarization_pipeline = DiarizationPipeline.from_pretrained(
            DIARIZATION_MODEL, token=hf_token
        )
        device = get_device()
        _diarization_pipeline.to(device)
    return _diarization_pipeline


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
def transcribe(audio_path: str) -> dict:
    """Run mlx-whisper on the audio file. Returns the full result dict."""
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=WHISPER_MODEL,
        word_timestamps=True,
        verbose=False,
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
    )
    return result


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------
def _resample_to_16k(audio_path: str) -> str:
    """Resample audio to 16kHz WAV to avoid pyannote chunk-size mismatches."""
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    # Downmix to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp.name, waveform, 16000)
    return tmp.name


def diarize(audio_path: str, num_speakers: int | None = None) -> list[dict]:
    """
    Run pyannote speaker diarization.
    Returns list of {start, end, speaker} dicts.
    """
    pipeline = _get_diarization_pipeline()

    # Resample to 16kHz to avoid sample-count mismatch errors
    resampled_path = _resample_to_16k(audio_path)

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    try:
        diarization = pipeline(resampled_path, **kwargs)
    finally:
        os.unlink(resampled_path)

    # pyannote ≥3.x returns DiarizeOutput; extract the Annotation
    if hasattr(diarization, "speaker_diarization"):
        diarization = diarization.speaker_diarization

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {"start": turn.start, "end": turn.end, "speaker": speaker}
        )
    return segments


# ---------------------------------------------------------------------------
# Merge: assign speakers to transcript words/segments
# ---------------------------------------------------------------------------
def _find_speaker(word_start: float, word_end: float, diar_segments: list[dict]) -> str:
    """Find the speaker for a word based on maximum overlap with diarization segments."""
    best_speaker = "Unknown"
    best_overlap = 0.0

    for seg in diar_segments:
        overlap_start = max(word_start, seg["start"])
        overlap_end = min(word_end, seg["end"])
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg["speaker"]

    return best_speaker


def merge_transcript_and_diarization(
    whisper_result: dict, diar_segments: list[dict]
) -> list[dict]:
    """
    Merge whisper word-level timestamps with diarization segments.
    Returns a list of utterances: [{speaker, start, end, text}, ...]
    """
    # Collect all words with their speaker assignments
    words_with_speakers = []
    for segment in whisper_result.get("segments", []):
        for word_info in segment.get("words", []):
            w_start = word_info["start"]
            w_end = word_info["end"]
            speaker = _find_speaker(w_start, w_end, diar_segments)
            words_with_speakers.append(
                {
                    "word": word_info["word"].strip(),
                    "start": w_start,
                    "end": w_end,
                    "speaker": speaker,
                }
            )

    if not words_with_speakers:
        return []

    # Group consecutive words by same speaker into utterances
    utterances = []
    current = {
        "speaker": words_with_speakers[0]["speaker"],
        "start": words_with_speakers[0]["start"],
        "end": words_with_speakers[0]["end"],
        "words": [words_with_speakers[0]["word"]],
    }

    for w in words_with_speakers[1:]:
        if w["speaker"] == current["speaker"]:
            current["end"] = w["end"]
            current["words"].append(w["word"])
        else:
            utterances.append(
                {
                    "speaker": current["speaker"],
                    "start": current["start"],
                    "end": current["end"],
                    "text": " ".join(current["words"]),
                }
            )
            current = {
                "speaker": w["speaker"],
                "start": w["start"],
                "end": w["end"],
                "words": [w["word"]],
            }

    # Don't forget the last utterance
    utterances.append(
        {
            "speaker": current["speaker"],
            "start": current["start"],
            "end": current["end"],
            "text": " ".join(current["words"]),
        }
    )

    return utterances


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    audio_path: str,
    num_speakers: int | None = None,
    progress_callback=None,
) -> dict:
    """
    Run the full transcription + diarization pipeline.

    Args:
        audio_path: Path to audio file.
        num_speakers: Optional hint for number of speakers.
        progress_callback: Optional callable(step: str, progress: float)

    Returns:
        {
            "utterances": [{speaker, start, end, text}, ...],
            "raw_transcript": str,
            "speakers": [str, ...],
        }
    """

    def _update(step, progress):
        if progress_callback:
            progress_callback(step, progress)

    _update("Transcribing audio...", 0.1)
    whisper_result = transcribe(audio_path)

    _update("Identifying speakers...", 0.5)
    diar_segments = diarize(audio_path, num_speakers=num_speakers)

    _update("Merging results...", 0.9)
    utterances = merge_transcript_and_diarization(whisper_result, diar_segments)

    speakers = sorted(set(u["speaker"] for u in utterances))

    _update("Done!", 1.0)

    return {
        "utterances": utterances,
        "raw_transcript": whisper_result.get("text", ""),
        "speakers": speakers,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_utterances(utterances: list[dict], speaker_names: dict | None = None) -> str:
    """
    Format utterances into a readable transcript string.

    Args:
        utterances: List of {speaker, start, end, text} dicts.
        speaker_names: Optional mapping of speaker IDs to custom names.
    """
    lines = []
    for u in utterances:
        name = (speaker_names or {}).get(u["speaker"], u["speaker"])
        ts = format_timestamp(u["start"])
        lines.append(f"[{ts}] {name}: {u['text']}")
    return "\n\n".join(lines)
