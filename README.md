# Whisper Subtitles

A Python application for generating and processing subtitles from audio files using the Whisper AI model and Gradio interface.

## Overview

This project transcribes audio files (mp3, wav, m4a) into subtitles using the `stable-ts` library, which is based on OpenAI's Whisper model. It provides a Gradio-based web interface to upload audio, select model size and language, and generate raw and arranged subtitles. Features include removing repeated words and merging subtitles into complete sentences, with support for Korean and other languages.

## Repository Structure

- `whisper_model.py`: Handles loading and cleaning up the Whisper model.
- `subtitle_processor.py`: Processes subtitles (removes duplicates, merges into sentences).
- `file_manager.py`: Manages file operations (directory setup, file saving, info extraction).
- `main.py`: Main script with the Gradio interface and core processing logic.
- `requirements.txt`: Lists required Python packages.

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git
- Google Colab (for running in the cloud) or a local environment with GPU support (recommended)

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/whisper-subtitles.git
   cd whisper-subtitles
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running in Google Colab
1. Open [Google Colab](https://colab.research.google.com).
2. Create a new notebook.
3. Add and run the following cells:

   **Clone the repository**:
   ```bash
   !git clone https://github.com/your-username/whisper-subtitles.git
   %cd whisper-subtitles
   ```

   **Install dependencies**:
   ```bash
   !pip install -r requirements.txt
   ```

   **Run the application**:
   ```python
   %run main.py
   ```

4. Access the Gradio interface via the provided URL (e.g., ngrok link).

### Running Locally
1. Ensure dependencies are installed (`pip install -r requirements.txt`).
2. Run the main script:
   ```bash
   python main.py
   ```
3. Open the Gradio interface URL in your browser.

## Usage
1. **Upload Audio**: Select an audio file (mp3, wav, m4a).
2. **Configure Settings**:
   - **Model Size**: Choose from `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`, `turbo` (larger models are more accurate but slower).
   - **Language**: Select `Auto`, `ko` (Korean), `en` (English), or `ja` (Japanese).
   - **Arrange Options**: Enable/disable removing repeated words and merging into complete sentences.
3. **Generate Subtitles**: Click "Generate Subtitles" to produce raw and arranged SRT files.
4. **Download Results**: Download raw or arranged subtitles as SRT files.

## Notes
- **Colab GPU**: For faster processing, use a GPU runtime in Colab (Runtime > Change runtime type > GPU).
- **Dependencies**: If `stable-ts` installation fails, try `pip install stable-ts==2.9.0` or install packages individually.
- **Gradio URL**: In Colab, you may need to allow pop-ups or manually copy the public URL to access the interface.

## Author
Made by 차유진

## License
MIT License