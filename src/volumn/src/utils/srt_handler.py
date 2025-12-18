"""
Utility functions for SRT subtitle handling
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SRTEntry:
    """Represents a single SRT subtitle entry"""
    index: int
    start_time: str  # Format: HH:MM:SS,mmm
    end_time: str    # Format: HH:MM:SS,mmm
    text: str
    
    def __str__(self) -> str:
        """Convert to SRT format string"""
        return f"{self.index}\n{self.start_time} --> {self.end_time}\n{self.text}\n"
    
    @property
    def start_seconds(self) -> float:
        """Get start time in seconds"""
        return timestamp_to_seconds(self.start_time)
    
    @property
    def end_seconds(self) -> float:
        """Get end time in seconds"""
        return timestamp_to_seconds(self.end_time)
    
    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        return self.end_seconds - self.start_seconds


def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert SRT timestamp to seconds.
    
    Args:
        timestamp: Timestamp in format HH:MM:SS,mmm
        
    Returns:
        Time in seconds
        
    Example:
        >>> timestamp_to_seconds("00:01:23,456")
        83.456
    """
    # Handle both comma and period as decimal separator
    timestamp = timestamp.replace(',', '.')
    
    match = re.match(r'(\d+):(\d+):(\d+)\.(\d+)', timestamp)
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    
    hours, minutes, seconds, milliseconds = match.groups()
    
    total_seconds = (
        int(hours) * 3600 +
        int(minutes) * 60 +
        int(seconds) +
        int(milliseconds) / 1000
    )
    
    return total_seconds


def seconds_to_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Timestamp in format HH:MM:SS,mmm
        
    Example:
        >>> seconds_to_timestamp(83.456)
        '00:01:23,456'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def parse_srt(srt_content: str) -> List[SRTEntry]:
    """
    Parse SRT content into list of SRTEntry objects.
    
    Args:
        srt_content: SRT file content as string
        
    Returns:
        List of SRTEntry objects
    """
    entries = []
    
    # Split by double newlines (entry separator)
    blocks = re.split(r'\n\s*\n', srt_content.strip())
    
    for block in blocks:
        if not block.strip():
            continue
        
        lines = block.strip().split('\n')
        
        if len(lines) < 3:
            continue
        
        try:
            # Parse index
            index = int(lines[0].strip())
            
            # Parse timestamps
            timestamp_line = lines[1].strip()
            match = re.match(r'(.+?)\s*-->\s*(.+)', timestamp_line)
            if not match:
                continue
            
            start_time = match.group(1).strip()
            end_time = match.group(2).strip()
            
            # Parse text (may be multiple lines)
            text = '\n'.join(lines[2:])
            
            entry = SRTEntry(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=text
            )
            entries.append(entry)
            
        except (ValueError, IndexError) as e:
            # Skip malformed entries
            continue
    
    return entries


def load_srt(srt_path: Path) -> List[SRTEntry]:
    """
    Load and parse SRT file.
    
    Args:
        srt_path: Path to SRT file
        
    Returns:
        List of SRTEntry objects
    """
    with open(srt_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    
    return parse_srt(content)


def save_srt(entries: List[SRTEntry], output_path: Path) -> None:
    """
    Save list of SRTEntry objects to file.
    
    Args:
        entries: List of SRTEntry objects
        output_path: Path to output SRT file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    srt_content = '\n'.join(str(entry) for entry in entries)
    
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write(srt_content)


def validate_srt(srt_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate SRT file format and content.
    
    Args:
        srt_path: Path to SRT file
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        entries = load_srt(srt_path)
        
        if not entries:
            errors.append("No valid entries found in SRT file")
            return False, errors
        
        # Check sequential indexing
        for i, entry in enumerate(entries, 1):
            if entry.index != i:
                errors.append(f"Entry {i}: Index mismatch (expected {i}, got {entry.index})")
        
        # Check timestamp validity
        for entry in entries:
            if entry.start_seconds >= entry.end_seconds:
                errors.append(
                    f"Entry {entry.index}: Start time >= end time "
                    f"({entry.start_time} --> {entry.end_time})"
                )
        
        # Check for overlaps
        for i in range(len(entries) - 1):
            current = entries[i]
            next_entry = entries[i + 1]
            
            if current.end_seconds > next_entry.start_seconds:
                errors.append(
                    f"Entry {current.index} and {next_entry.index}: "
                    f"Overlapping timestamps"
                )
        
        # Check for empty text
        for entry in entries:
            if not entry.text.strip():
                errors.append(f"Entry {entry.index}: Empty text")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        errors.append(f"Failed to validate SRT: {str(e)}")
        return False, errors


def get_srt_statistics(srt_path: Path) -> Dict[str, Any]:
    """
    Get statistics about an SRT file.
    
    Args:
        srt_path: Path to SRT file
        
    Returns:
        Dictionary with statistics
    """
    entries = load_srt(srt_path)
    
    if not entries:
        return {
            'total_entries': 0,
            'total_duration': 0.0,
            'average_duration': 0.0,
            'total_words': 0,
            'average_words_per_entry': 0.0
        }
    
    total_duration = entries[-1].end_seconds
    durations = [entry.duration for entry in entries]
    word_counts = [len(entry.text.split()) for entry in entries]
    
    # Count non-dialogue events
    non_dialogue_count = sum(
        1 for entry in entries if entry.text.strip().startswith('[[')
    )
    
    return {
        'total_entries': len(entries),
        'dialogue_entries': len(entries) - non_dialogue_count,
        'non_dialogue_entries': non_dialogue_count,
        'total_duration': total_duration,
        'average_duration': sum(durations) / len(durations),
        'min_duration': min(durations),
        'max_duration': max(durations),
        'total_words': sum(word_counts),
        'average_words_per_entry': sum(word_counts) / len(word_counts),
        'total_characters': sum(len(entry.text) for entry in entries)
    }


def merge_srt_entries(entries: List[SRTEntry], max_gap: float = 0.5) -> List[SRTEntry]:
    """
    Merge consecutive SRT entries that are close together.
    
    Args:
        entries: List of SRTEntry objects
        max_gap: Maximum gap in seconds to allow merging
        
    Returns:
        List of merged SRTEntry objects
    """
    if not entries:
        return []
    
    merged = []
    current = None
    
    for entry in entries:
        if current is None:
            current = entry
            continue
        
        gap = entry.start_seconds - current.end_seconds
        
        if gap <= max_gap:
            # Merge with current entry
            current = SRTEntry(
                index=current.index,
                start_time=current.start_time,
                end_time=entry.end_time,
                text=current.text + ' ' + entry.text
            )
        else:
            # Save current and start new entry
            merged.append(current)
            current = entry
    
    # Add the last entry
    if current is not None:
        merged.append(current)
    
    # Reindex
    for i, entry in enumerate(merged, 1):
        entry.index = i
    
    return merged


def split_long_entries(
    entries: List[SRTEntry],
    max_chars: int = 84,
    max_duration: float = 7.0
) -> List[SRTEntry]:
    """
    Split long SRT entries into smaller ones.
    
    Args:
        entries: List of SRTEntry objects
        max_chars: Maximum characters per entry
        max_duration: Maximum duration per entry in seconds
        
    Returns:
        List of split SRTEntry objects
    """
    result = []
    
    for entry in entries:
        if len(entry.text) <= max_chars and entry.duration <= max_duration:
            result.append(entry)
            continue
        
        # Split text by sentences or phrases
        words = entry.text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(' '.join(current_chunk)) > max_chars:
                if len(current_chunk) > 1:
                    current_chunk.pop()
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                else:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Create new entries with proportional timing
        duration_per_char = entry.duration / len(entry.text)
        current_time = entry.start_seconds
        
        for chunk in chunks:
            chunk_duration = len(chunk) * duration_per_char
            
            new_entry = SRTEntry(
                index=0,  # Will be reindexed later
                start_time=seconds_to_timestamp(current_time),
                end_time=seconds_to_timestamp(current_time + chunk_duration),
                text=chunk
            )
            result.append(new_entry)
            current_time += chunk_duration
    
    # Reindex
    for i, entry in enumerate(result, 1):
        entry.index = i
    
    return result


def merge_close_subtitles(
    entries: List[SRTEntry],
    min_gap_ms: float = 200,
    max_merged_duration: float = 10.0
) -> List[SRTEntry]:
    """
    Merge subtitle entries that are very close together to improve TTS quality
    and reduce audio popping artifacts.
    
    Args:
        entries: List of SRTEntry objects
        min_gap_ms: Minimum gap in milliseconds; entries closer than this will be merged
        max_merged_duration: Maximum duration of merged entry in seconds
        
    Returns:
        List of merged SRTEntry objects
    """
    if not entries:
        return []
    
    result = []
    current_merge = None
    
    for entry in entries:
        if current_merge is None:
            # Start new merge group
            current_merge = SRTEntry(
                index=entry.index,
                start_time=entry.start_time,
                end_time=entry.end_time,
                text=entry.text
            )
        else:
            # Calculate gap to previous entry
            gap_ms = (entry.start_seconds - current_merge.end_seconds) * 1000
            merged_duration = entry.end_seconds - current_merge.start_seconds
            
            # Merge if gap is small and merged duration is acceptable
            if gap_ms < min_gap_ms and merged_duration <= max_merged_duration:
                # Merge: extend end time and append text
                current_merge.end_time = entry.end_time
                # Add a space between merged texts
                current_merge.text = current_merge.text.strip() + ' ' + entry.text.strip()
            else:
                # Gap is large enough or would exceed max duration - save current and start new
                result.append(current_merge)
                current_merge = SRTEntry(
                    index=entry.index,
                    start_time=entry.start_time,
                    end_time=entry.end_time,
                    text=entry.text
                )
    
    # Don't forget the last merge group
    if current_merge is not None:
        result.append(current_merge)
    
    # Reindex
    for i, entry in enumerate(result, 1):
        entry.index = i
    
    return result
