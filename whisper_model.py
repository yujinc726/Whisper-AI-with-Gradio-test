import torch
import stable_whisper
import whisper

# Global variables to store the current model
current_model = None
current_model_size = None
current_stable_ts = None

def load_model(model_size, stable_ts):
    """Load or reload the Whisper model based on the selected model size."""
    global current_model, current_model_size, current_stable_ts
    if current_model is not None and not(current_model_size == model_size and current_stable_ts == stable_ts):
         # Clear the existing model from memory
        current_model = None
        torch.cuda.empty_cache()  # Clear GPU memory if applicable
    if current_model is None:
        print(f"Loading Whisper model: {model_size} {'(stable-ts)' if stable_ts else ''}")
    if stable_ts:
        current_model = stable_whisper.load_model(model_size)
    else:
        current_model = whisper.load_model(model_size)
    current_model_size = model_size
    current_stable_ts = stable_ts
    return current_model

def cleanup_model():
    """Clean up the current model from memory."""
    global current_model
    if current_model is not None:
        del current_model
        torch.cuda.empty_cache()