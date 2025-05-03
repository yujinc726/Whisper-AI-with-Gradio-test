import gradio as gr
import subprocess
import sys
from whisper_model import load_model, cleanup_model
from subtitle_processor import arrange_subtitles
from file_manager import setup_directories, save_uploaded_file, get_file_info

def install_package(use_stable_ts):
    """Install the required package based on the use_stable_ts flag."""
    package_name = "stable-ts" if use_stable_ts else "openai-whisper"
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_name}: {str(e)}")
        raise

def transcribe(file_info, decode_options, model):
    """Transcribe the audio file using the Whisper model."""
    print(f'Transcribing "{file_info["file_name"]}{file_info["file_extension"]}"...')
    result = model.transcribe(file_info["file_path"], **decode_options)
    print(f'Finished transcribing.')
    return result

def process_audio(audio_file, language, remove_repeated, merge, model_size, use_stable_ts):
    """Process the uploaded audio file and generate subtitles."""
    if not audio_file:
        return None, None, None, None

    try:
        # Install the required package
        install_package(use_stable_ts)

        upload_dir = setup_directories()
        file_info = save_uploaded_file(audio_file, upload_dir)
        if not file_info:
            return None, None, None, None

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

        # Load the selected model
        model = load_model(model_size, use_stable_ts)

        # Transcribe
        subtitles = transcribe(file_info, decode_options, model)
        suffix = "stable-ts" if use_stable_ts else "whisper"
        subtitles_path = f'download/{file_info["file_name"]}_{model_size}_{suffix}_word_ts.srt'
        subtitles.to_srt_vtt(subtitles_path, **srt_options)

        # Read raw subtitles content
        with open(subtitles_path, 'r', encoding='utf-8') as f:
            raw_subtitles_content = f.read()

        # Arrange subtitles
        arranged_subtitles = arrange_subtitles(subtitles_path, **arrange_options)
        arranged_path = f'download/{file_info["file_name"]}_{model_size}_{suffix}_word_ts_arranged.srt'
        with open(arranged_path, 'w', encoding='utf-8') as f:
            f.writelines(arranged_subtitles)

        # Read arranged subtitles content
        with open(arranged_path, 'r', encoding='utf-8') as f:
            arranged_subtitles_content = f.read()

        return (
            raw_subtitles_content,
            subtitles_path,
            arranged_subtitles_content,
            arranged_path
        )
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        return None, None, None, None

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
            audio_input = gr.File(label="Upload Audio File (mp3, wav, m4a)")
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
            use_stable_ts = gr.Checkbox(label="Use Stable-TS", value=True)
            with gr.Accordion(label="Arrange Options", open=False):
                remove_repeated = gr.Checkbox(label="Remove Repeated Words", value=True)
                merge = gr.Checkbox(label="Merge into Complete Sentences", value=True)
            submit_btn = gr.Button("Generate Subtitles")
        
        # Right column for outputs
        with gr.Column(scale=1):
            gr.Markdown("#### Raw Subtitles")
            raw_subtitles_text = gr.Textbox(
                label="Raw Subtitles Content",
                lines=10,
                max_lines=10,
                interactive=True,
                elem_classes="textbox-fixed"
            )
            raw_subtitles = gr.DownloadButton(label="Download Raw Subtitles (SRT)")
            gr.Markdown("#### Arranged Subtitles")
            arranged_subtitles_text = gr.Textbox(
                label="Arranged Subtitles Content",
                lines=10,
                max_lines=10,
                interactive=True,
                elem_classes="textbox-fixed"
            )
            arranged_subtitles = gr.DownloadButton(label="Download Arranged Subtitles (SRT)")

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, language, remove_repeated, merge, model_size, use_stable_ts],
        outputs=[raw_subtitles_text, raw_subtitles, arranged_subtitles_text, arranged_subtitles]
    )

# Launch Gradio interface
demo.launch(debug=True)

# Clean up model
cleanup_model()