import torch
import subprocess
import sys

# Global variables to store the current model
current_model = None
current_model_size = None
current_use_stable_ts = None

def install_package(package_name):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def load_model(model_size, use_stable_ts):
    """Load or reload the Whisper model based on the selected model size and package."""
    global current_model, current_model_size, current_use_stable_ts
    
    # Install the appropriate package if not already installed
    if use_stable_ts:
        try:
            import stable_whisper
        except ImportError:
            print("Installing stable-ts...")
            install_package("stable-ts")
            import stable_whisper
    else:
        try:
            import whisper
        except ImportError:
            print("Installing openai-whisper...")
            install_package("openai-whisper")
            import whisper

    # Clear the existing model if model size or package type has changed
    if (current_model is not None and 
        (current_model_size != model_size or current_use_stable_ts != use_stable_ts)):
        del current_model
        torch.cuda.empty_cache()  # Clear GPU memory if applicable
        current_model = None

    if current_model_size != model_size or current_use_stable_ts != use_stable_ts:
        print(f"Loading {'stable-ts' if use_stable_ts else 'openai-whisper'} model: {model_size}")
        if use_stable_ts:
            current_model = stable_whisper.load_model(model_size)
        else:
            current_model = whisper.load_model(model_size)
        current_model_size = model_size
        current_use_stable_ts = use_stable_ts
    
    return current_model

def cleanup_model():
    """Clean up the current model from memory."""
    global current_model
    if current_model is not None:
        del current_model
        torch.cuda.empty_cache()
        current_model = None