import torch
import stable_whisper

# Global variables to store the current model
current_model = None
current_model_size = None

def load_model(model_size):
    """Load or reload the Whisper model based on the selected model size."""
    global current_model, current_model_size
    if current_model is not None and current_model_size != model_size:
        # Clear the existing model from memory
        del current_model
        torch.cuda.empty_cache()  # Clear GPU memory if applicable
    if current_model_size != model_size:
        print(f"Loading Whisper model: {model_size}")
        current_model = stable_whisper.load_model(model_size)
        current_model_size = model_size
    return current_model

def cleanup_model():
    """Clean up the current model from memory."""
    global current_model
    if current_model is not None:
        del current_model
        torch.cuda.empty_cache()