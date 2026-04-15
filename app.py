"""
Gradio UI for Transcript — local audio transcription with speaker diarization.
"""

import gradio as gr
from pipeline import run_pipeline, format_utterances, format_timestamp


def process_audio(audio_path, num_speakers, progress=gr.Progress()):
    """Main handler for the Gradio interface."""
    if audio_path is None:
        raise gr.Error("Please upload an audio file.")

    num_spk = int(num_speakers) if num_speakers and num_speakers > 0 else None

    def progress_callback(step, pct):
        progress(pct, desc=step)

    result = run_pipeline(
        audio_path,
        num_speakers=num_spk,
        progress_callback=progress_callback,
    )

    utterances = result["utterances"]
    speakers = result["speakers"]

    # Build formatted transcript
    transcript_text = format_utterances(utterances)

    # Build per-speaker breakdown
    speaker_summary = f"**{len(speakers)} speaker(s) detected:** {', '.join(speakers)}"

    # Build a dataframe-friendly list for the table view
    table_data = [
        [format_timestamp(u["start"]), format_timestamp(u["end"]), u["speaker"], u["text"]]
        for u in utterances
    ]

    return transcript_text, speaker_summary, table_data


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
css = """
.transcript-output textarea {
    font-family: 'SF Mono', 'Menlo', 'Monaco', monospace !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}
"""

with gr.Blocks(title="Transcript") as demo:
    gr.Markdown(
        """
        # Transcript
        **Local audio transcription with speaker diarization.**
        Powered by mlx-whisper + pyannote · runs entirely on your Mac.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Upload Audio",
                type="filepath",
                sources=["upload", "microphone"],
            )
            num_speakers = gr.Number(
                label="Number of Speakers (optional)",
                value=0,
                minimum=0,
                maximum=20,
                step=1,
                info="Set to 0 for automatic detection",
            )
            transcribe_btn = gr.Button(
                "Transcribe", variant="primary", size="lg"
            )

        with gr.Column(scale=2):
            speaker_info = gr.Markdown(label="Speakers")
            transcript_output = gr.Textbox(
                label="Transcript",
                lines=20,
                max_lines=50,
                elem_classes=["transcript-output"],
            )

    with gr.Accordion("Detailed View", open=False):
        table_output = gr.Dataframe(
            headers=["Start", "End", "Speaker", "Text"],
            label="Segments",
            wrap=True,
        )

    transcribe_btn.click(
        fn=process_audio,
        inputs=[audio_input, num_speakers],
        outputs=[transcript_output, speaker_info, table_output],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft(), css=css)
