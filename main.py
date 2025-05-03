import gradio as gr
from whisper_model import load_model, cleanup_model
from subtitle_processor import arrange_subtitles
from file_manager import setup_directories, save_uploaded_file, get_file_info
import os
import uuid

def transcribe(file_info, decode_options, model):
    """Transcribe the audio file using the Whisper model."""
    print(f'Transcribing "{file_info["file_name"]}{file_info["file_extension"]}"...')
    result = model.transcribe(file_info["file_path"], **decode_options)
    print(f'Finished transcribing.')
    return result

def process_audio(audio_files, language, remove_repeated, merge, model_size):
    """Process multiple uploaded audio files and generate subtitles."""
    if not audio_files:
        return [gr.update(visible=False)] * 4
    
    try:
        upload_dir = setup_directories()
        outputs = []
        
        # Load the selected model once
        model = load_model(model_size)
        
        decode_options = {
            "task": "transcribe",
            "fp16": True,
            "language": language if language != "Auto" else None,
            "verbose": False,
            "word_timestamps": True
        }
        srt_options = {
            "segment_level": False,
            "word_level": True
        }
        arrange_options = {
            "remove_repeated": remove_repeated,
            "merge": merge
        }

        for audio_file in audio_files:
            file_info = save_uploaded_file(audio_file, upload_dir)
            if not file_info:
                continue

            # Transcribe
            subtitles = transcribe(file_info, decode_options, model)
            raw_subtitles_path = f'download/{file_info["file_name"]}_{model_size}_stable-ts_word_ts.srt'
            subtitles.to_srt_vtt(raw_subtitles_path, **srt_options)

            # Read raw subtitles content
            with open(raw_subtitles_path, 'r', encoding='utf-8') as f:
                raw_subtitles_content = f.read()

            # Arrange subtitles
            arranged_subtitles = arrange_subtitles(raw_subtitles_path, **arrange_options)
            arranged_path = f'download/{file_info["file_name"]}_{model_size}_stable-ts_word_ts_arranged.srt'
            with open(arranged_path, 'w', encoding='utf-8') as f:
                f.writelines(arranged_subtitles)

            # Read arranged subtitles content
            with open(arranged_path, 'r', encoding='utf-8') as f:
                arranged_subtitles_content = f.read()

            # Create unique component IDs
            raw_text_id = f"raw_text_{uuid.uuid4()}"
            arranged_text_id = f"arranged_text_{uuid.uuid4()}"

            # Create output components for this file
            file_outputs = [
                gr.Markdown(f"### Results for {file_info['file_name']}{file_info['file_extension']}"),
                gr.Textbox(
                    label="Raw Subtitles Content",
                    value=raw_subtitles_content,
                    lines=10,
                    max_lines=10,
                    interactive=True,
                    elem_classes="textbox-fixed",
                    elem_id=raw_text_id
                ),
                gr.DownloadButton(
                    label="Download Raw Subtitles (SRT)",
                    value=raw_subtitles_path
                ),
                gr.Textbox(
                    label="Arranged Subtitles Content",
                    value=arranged_subtitles_content,
                    lines=10,
                    max_lines=10,
                    interactive=True,
                    elem_classes="textbox-fixed",
                    elem_id=arranged_text_id
                ),
                gr.DownloadButton(
                    label="Download Arranged Subtitles (SRT)",
                    value=arranged_path
                ),
                gr.Markdown("---")
            ]
            outputs.extend(file_outputs)

        return outputs

    except Exception as e:
        print(f"Error processing audio files: {str(e)}")
        return [gr.update(visible=False)] * 4

# Custom CSS to control Textbox size
custom_css = """
.textbox-fixed {
    height: 200px !important;
    max-height: 200px !important;
    overflow-y: auto !important;
}
"""

# Gradio Interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Whisper AI with Gradio")
    gr.Markdown("## 자막 생성 프로그램")
    gr.Markdown("Made by 차유진")
    
    with gr.Row():
        # Left column for settings and upload
        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            audio_input = gr.Files(label="Upload Audio Files (mp3, wav, m4a)", file_types=[".mp3", ".wav", ".m4a"])
            with gr.Row():
                model_size = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"],
                    value="large-v2",
                    label="Model Size"
                )
                language = gr.Dropdown(
                    choices=["Auto", "ko", "en", "ja"],
                    value="Auto",
                    label="Language"
                )
            with gr.Accordion(label="Arrange Options", open=False):
                remove_repeated = gr.Checkbox(label="Remove Repeated Words", value=True)
                merge = gr.Checkbox(label="Merge into Complete Sentences", value=True)
            submit_btn = gr.Button("Generate Subtitles")
        
        # Right column for outputs
        with gr.Column(scale=1):
            output_components = gr.Blocks()

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, language, remove_repeated, merge, model_size],
        outputs=output_components
    )

# Launch Gradio interface
demo.launch(debug=True)

# Clean up model
cleanup_model()