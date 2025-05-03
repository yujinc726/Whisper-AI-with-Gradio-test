import os
import unicodedata
from pathlib import Path
import shutil

def setup_directories():
    """Set up directories for uploads and downloads."""
    upload_dir = "/content/uploads/"
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs("download", exist_ok=True)
    return upload_dir

def get_file_info(file_path):
    """Extract file information from the provided file path."""
    normalized_file = unicodedata.normalize('NFC', os.path.basename(file_path))
    file_name = Path(normalized_file).stem
    file_extension = Path(normalized_file).suffix
    file_info = {"file_path": file_path, "file_name": file_name, "file_extension": file_extension}
    return file_info

def save_uploaded_file(audio_file, upload_dir):
    """Save uploaded file to temporary directory and return file info."""
    valid_extensions = ['.mp3', '.wav', '.m4a']
    if not any(audio_file.name.lower().endswith(ext) for ext in valid_extensions):
        return None
    
    # 원본 파일명 유지
    original_file_name = Path(audio_file.name).name
    file_path = os.path.join(upload_dir, original_file_name)
    
    shutil.copy(audio_file.name, file_path)
    return get_file_info(file_path)