import torch
import importlib

# Global variables to store the current model
current_model = None
current_model_size = None
current_model_type = None

def load_model(model_size, use_stable_ts):
    """Load or reload the Whisper model based on the selected model size and type."""
    global current_model, current_model_size, current_model_type
    if current_model is not None and (current_model_size != model_size or current_model_type != use_stable_ts):
        # Clear the existing model from memory
        del current_model
        torch.cuda.empty_cache()  # Clear GPU memory if applicable
    if current_model_size != model_size or current_model_type != use_stable_ts:
        print(f"Loading {'Stable-TS' if use_stable_ts else 'Whisper'} model: {model_size}")
        module_name = "stable_whisper" if use_stable_ts else "whisper"
        module = importlib.import_module(module_name)
        current_model = module.load_model(model_size)
        current_model_size = model_size
        current_model_type = use_stable_ts
    return current_model

def cleanup_model():
    """Clean up the current model from memory."""
    global current_model
    if current_model is not None:
        del current_model
        torch.cuda.empty_cache()