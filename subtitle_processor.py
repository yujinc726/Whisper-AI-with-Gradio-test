import re

def remove_repeated_words(subtitle_lines):
    """Remove repeated subtitle lines."""
    filtered_lines = []
    previous_text = ""
    skip_next_lines = 0

    for i in range(len(subtitle_lines)):
        if skip_next_lines > 0:
            skip_next_lines -= 1
            continue
        if re.match(r"^\d+$", subtitle_lines[i].strip()):
            if i + 1 < len(subtitle_lines):
                next_line = subtitle_lines[i + 1].strip()
                if re.match(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", next_line):
                    if i + 2 < len(subtitle_lines):
                        current_text = subtitle_lines[i + 2].strip()
                        if current_text != previous_text:
                            filtered_lines.append(subtitle_lines[i])
                            filtered_lines.append(subtitle_lines[i + 1])
                            filtered_lines.append(subtitle_lines[i + 2])
                            filtered_lines.append('\n')
                            previous_text = current_text
                        skip_next_lines = 3
    return filtered_lines

def is_complete_sentence(text):
    """Check if the text is a complete sentence."""
    return text.strip().endswith(('.', '!', '?', '니다', '어요', '에요', '예요', '구요', '고요', '죠'))

def merge_words(subtitle_lines):
    """Merge subtitle lines into complete sentences."""
    merged_subtitles = []
    current_subtitle = ""
    current_start_time = None
    current_end_time = ""
    subtitle_index = 1
    skip_next_lines = 0

    for i in range(len(subtitle_lines)):
        if skip_next_lines > 0:
            skip_next_lines -= 1
            continue
        if i + 1 < len(subtitle_lines):
            timestamp_match = re.match(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", subtitle_lines[i + 1])
            if subtitle_lines[i].strip().isdigit() and timestamp_match and i + 2 < len(subtitle_lines):
                start_time, end_time = timestamp_match.groups()
                if current_subtitle and is_complete_sentence(current_subtitle):
                    merged_subtitles.append(f"{subtitle_index}\n{current_start_time} --> {current_end_time}\n{current_subtitle.strip()}\n\n")
                    subtitle_index += 1
                    current_subtitle = ""
                    current_start_time = start_time
                if current_start_time is None:
                    current_start_time = start_time
                current_end_time = end_time
                current_subtitle += subtitle_lines[i + 2].strip() + " "
                skip_next_lines = 3

    if current_subtitle:
        merged_subtitles.append(f"{subtitle_index}\n{current_start_time} --> {current_end_time}\n{current_subtitle.strip()}\n")
    return merged_subtitles

def arrange_subtitles(subtitles_path, remove_repeated, merge):
    """Arrange subtitles by removing repeats and merging if specified."""
    print('Arranging subtitles...')
    with open(subtitles_path, 'r', encoding='utf-8') as f:
        arranged_subtitles = f.readlines()
    if remove_repeated:
        arranged_subtitles = remove_repeated_words(arranged_subtitles)
    if merge:
        arranged_subtitles = merge_words(arranged_subtitles)
    print('Finished arranging subtitles.')
    return arranged_subtitles