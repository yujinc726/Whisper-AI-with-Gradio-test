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
        return {"components": [], "visible": False}
    
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

            # Store output data for this file
            outputs.append({
                "title": f"Results for {file_info['file_name']}{file_info['file_extension']}",
                "raw_subtitles": raw_subtitles_content,
                "raw_path": raw_subtitles_path,
                "arranged_subtitles": arranged_subtitles_content,
                "arranged_path": arranged_path
            })

        return {"components": outputs, "visible": True}

    except Exception as e:
        print(f"Error processing audio files: {str(e)}")
        return {"components": [], "visible": False}

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
            output_state = gr.State({"components": [], "visible": False})
            with gr.Column(visible=False) as output_column:
                # Create multiple output sets (up to 10 files for example)
                output_components = []
                for i in range(10):  # Support up to 10 files
                    with gr.Group(visible=False) as file_output:
                        title = gr.Markdown()
                        raw_text = gr.Textbox(
                            label="Raw Subtitles Content",
                            lines=10,
                            max_lines=10,
                            interactive=True,
                            elem_classes="textbox-fixed"
                        )
                        raw_download = gr.DownloadButton(label="Download Raw Subtitles (SRT)")
                        arranged_text = gr.Textbox(
                            label="Arranged Subtitles Content",
                            lines=10,
                            max_lines=10,
                            interactive=True,
                            elem_classes="textbox-fixed"
                        )
                        arranged_download = gr.DownloadButton(label="Download Arranged Subtitles (SRT)")
                        separator = gr.Markdown("---")
                    output_components.append({
                        "group": file_output,
                        "title": title,
                        "raw_text": raw_text,
                        "raw_download": raw_download,
                        "arranged_text": arranged_text,
                        "arranged_download": arranged_download,
                        "separator": separator
                    })

    def update_outputs(state):
        """Update output components based on state."""
        updates = []
        for i, comp in enumerate(output_components):
            if i < len(state["components"]):
                file_data = state["components"][i]
                updates.extend([
                    gr.update(value=file_data["title"], visible=True),  # title
                    gr.update(value=file_data["raw_subtitles"], visible=True),  # raw_text
                    gr.update(value=file_data["raw_path"], visible=True),  # raw_download
                    gr.update(value=file_data["arranged_subtitles"], visible=True),  # arranged_text
                    gr.update(value=file_data["arranged_path"], visible=True),  # arranged_download
                    gr.update(visible=True),  # separator
                    gr.update(visible=True)  # group
                ])
            else:
                updates.extend([
                    gr.update(visible=False),  # title
                    gr.update(visible=False),  # raw_text
                    gr.update(visible=False),  # raw_download
                    gr.update(visible=False),  # arranged_text
                    gr.update(visible=False),  # arranged_download
                    gr.update(visible=False),  # separator
                    gr.update(visible=False)  # group
                ])
        updates.append(gr.update(visible=len(state["components"]) > 0))  # output_column
        return updates

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, language, remove_repeated, merge, model_size],
        outputs=[output_state],
        _js="""
        async (audio_files, language, remove_repeated, merge, model_size) => {
            const result = await gradioApp().callFunction(
                null,
                audio_files,
                language,
                remove_repeated,
                merge,
                model_size
            );
            return result;
        }
        """
    ).then(
        fn=update_outputs,
        inputs=[output_state],
        outputs=[comp for c in output_components for comp in [
            c["title"],
            c["raw_text"],
            c["raw_download"],
            c["arranged_text"],
            c["arranged_download"],
            c["separator"],
            c["group"]
        ]] + [output_column]
    )

# Launch Gradio interface
demo.launch(debug=True)

# Clean up model
cleanup_model()