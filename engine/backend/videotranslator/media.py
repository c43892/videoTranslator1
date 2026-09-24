import json
import math
import os
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from .domain import NeedsReview

RATE = 48000
DEMUXERS = 'mov,matroska,webm,avi,mpegts,mpeg,flv,asf,ogg,wav,flac,mp3'

def run(args, timeout=1800):
    if timeout == 1800 and os.getenv('CLOUD_HEALTH_EXECUTION') == '1':
        timeout = None  # Parent supervises CPU/I/O activity and owns cancellation.
    proc = subprocess.run(args, capture_output=True, timeout=timeout)
    if proc.returncode:
        # ffmpeg can print hostile filenames; bound error size.
        raise RuntimeError(proc.stderr.decode('utf-8', errors='replace')[-1200:])
    return proc.stdout

def probe(path):
    raw = run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe',
               '-format_whitelist', DEMUXERS, '-show_format', '-show_streams', '-of', 'json', str(path)], timeout=30)
    return json.loads(raw)

def ffmpeg(args):
    run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y'] + [str(x) for x in args])

def input_args(path):
    return ['-protocol_whitelist', 'file,pipe', '-format_whitelist', DEMUXERS, '-i', path]

def extract_audio(video, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(input_args(video) + ['-map', '0:a:0', '-vn', '-ar', RATE, '-ac', 2, '-c:a', 'flac', output])

def canonical_audio(source, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(input_args(source) + ['-ar', RATE, '-ac', 2, '-c:a', 'pcm_s16le', output])

def clip(source, output, start, end):
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(input_args(source) + ['-ss', f'{start:.6f}', '-t', f'{end-start:.6f}', '-ar', RATE, '-ac', 1,
                                 '-c:a', 'pcm_s16le', output])

def quality(path):
    data, rate = sf.read(path, always_2d=True)
    rms = float(np.sqrt(np.mean(data**2))) if len(data) else 0
    return {'duration': len(data)/rate, 'rms': rms,
            'clipped_fraction': float(np.mean(np.abs(data) >= 0.999)) if len(data) else 1}

def align(source, output, window, max_speedup):
    output.parent.mkdir(parents=True, exist_ok=True)
    # max_speedup remains a compatibility argument, not a rejection threshold.
    duration = sf.info(source).duration
    if not math.isfinite(window) or window <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Audio duration and alignment window must be positive and finite')
    ratio = duration / window
    filters = []
    if ratio > 1:
        # Each stage stays <= 2: larger single atempo factors may skip samples.
        stages = max(1, math.ceil(math.log2(ratio)))
        factor = ratio ** (1 / stages)
        filters.extend([f'atempo={factor:.12f}'] * stages)
    # Trim after resampling: trimming at the TTS sample rate can add more than
    # one 48 kHz sample when converted, especially with Whisper float timestamps.
    frames = max(1, round(window * RATE))
    filters.extend([f'aresample={RATE}', f'apad=whole_len={frames}', f'atrim=end_sample={frames}'])
    ffmpeg(input_args(source) + ['-af', ','.join(filters), '-ar', RATE, '-ac', 2, '-c:a', 'pcm_s16le', output])

# Five milliseconds at 48 kHz: smooth edit discontinuities without moving speech.
SPLICE_FADE_FRAMES = round(.005 * RATE)


def _edge_envelope(position, size, start, end):
    """Raised-cosine gain inside [start, end), zero outside; bounded for tiny clips."""
    indices = np.arange(position, position + size)
    gain = ((indices >= start) & (indices < end)).astype(np.float32)
    fade = min(SPLICE_FADE_FRAMES, (end - start) // 2)
    if fade <= 1:
        # A one/two-sample event cannot have an interior and two smooth edges.
        return np.zeros(size, dtype=np.float32)
    distance = np.minimum(indices - start, end - 1 - indices)
    ramp = np.clip(distance / (fade - 1), 0, 1)
    return gain * (.5 - .5 * np.cos(np.pi * ramp)).astype(np.float32)


def _merge_ranges(ranges):
    merged = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _audible_bounds(audio, count):
    """Find exact digital padding so the fade lands on speech, not padded silence."""
    first, last = 0, count
    while first < last:
        audio.seek(first)
        data = audio.read(min(RATE, last-first), dtype='float32', always_2d=True)
        active = np.flatnonzero(np.any(data != 0, axis=1))
        if len(active):
            first += int(active[0]); break
        first += len(data)
        if not len(data):
            return count, count
    while last > first:
        offset = max(first, last-RATE)
        audio.seek(offset)
        data = audio.read(last-offset, dtype='float32', always_2d=True)
        active = np.flatnonzero(np.any(data != 0, axis=1))
        if len(active):
            last = offset + int(active[-1]) + 1; break
        last = offset
    return first, last


def dialogue_timeline(segments, storage, output, duration, original=None, preserve_intervals=None):
    """Replace dubbed windows with 5 ms edge blends; keep absolute sample positions."""
    frames = round(duration * RATE)
    events = preserve_intervals or []
    if original is None and (events or any(s.render_status == 'original' for s in segments)):
        raise ValueError('Original dialogue is required for fallback/events')
    translated = []
    for segment in segments:
        if segment.render_status != 'translated':
            continue
        info = sf.info(storage.path(segment.aligned_audio))
        if info.samplerate != RATE or info.channels != 2:
            raise ValueError('Aligned speech must be canonical stereo audio')
        offset = round(segment.start * RATE)
        if offset < 0 or offset + info.frames > frames + 2:
            raise ValueError('Dialogue segment lies outside the video timeline')
        translated.append((segment, offset, min(info.frames, frames-offset)))
    # Merge adjacent/overlapping replacements: do not leak original words back
    # into each internal join, or repeatedly attenuate overlapping windows.
    replaced = _merge_ranges([(offset, offset+count) for _, offset, count in translated])
    preserved = [(s.start, s.end) for s in segments if s.render_status == 'original']
    preserved += [(e['start'], e['end']) for e in events]
    keep = []
    for start, end in preserved:
        offset, last = round(start*RATE), round(end*RATE)
        if offset < 0 or last > frames+2 or last < offset:
            raise ValueError('Invalid original fallback/event interval')
        keep.append((offset, min(last, frames)))
    keep = _merge_ranges(keep)
    output.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(output, 'w+', samplerate=RATE, channels=2, subtype='FLOAT') as out:
        # Stream one-second blocks to keep memory bounded for long videos.
        source = sf.SoundFile(original) if original is not None else None
        try:
            if source and (source.samplerate != RATE or source.channels != 2):
                raise ValueError('Original dialogue must be canonical stereo audio')
            for position in range(0, frames, RATE):
                size = min(RATE, frames-position)
                data = np.zeros((size, 2), dtype=np.float32)
                if source:
                    chunk = source.read(size, dtype='float32', always_2d=True)
                    data[:len(chunk)] = chunk
                    gain = np.ones(size, dtype=np.float32)
                    for start, end in replaced:
                        if start < position+size and end > position:
                            gain = np.minimum(gain, 1-_edge_envelope(position, size, start, end))
                    for start, end in keep:
                        if start < position+size and end > position:
                            gain = np.maximum(gain, _edge_envelope(position, size, start, end))
                    data *= gain[:, None]
                out.write(data)
        finally:
            if source:
                source.close()
        for segment, offset, count in translated:
            with sf.SoundFile(storage.path(segment.aligned_audio)) as dubbed:
                first, last = _audible_bounds(dubbed, count)
                dubbed.seek(0)
                position = 0
                while position < count:
                    data = dubbed.read(min(RATE, count-position), dtype='float32', always_2d=True)
                    if not len(data):
                        raise ValueError('Synthesized audio is shorter than declared')
                    data *= _edge_envelope(position, len(data), first, last)[:, None]
                    out.seek(offset+position)
                    previous = out.read(len(data), dtype='float32', always_2d=True)
                    out.seek(offset+position); out.write(previous+data)
                    position += len(data)


def mix(dialogue, music, effects, output, duration):
    ffmpeg(input_args(dialogue) + input_args(music) + input_args(effects) + [
        '-filter_complex', '[0:a][1:a][2:a]amix=inputs=3:duration=first:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11[a]',
        '-map', '[a]', '-t', f'{duration:.6f}', '-ar', RATE, '-ac', 2, '-c:a', 'pcm_s16le', output])

def assemble(video, audio, output, metadata, subtitle_path=None, subtitle_language='en'):
    video_stream = next(s for s in metadata['streams'] if s['codec_type'] == 'video')
    copy = video_stream.get('codec_name') == 'h264' and video_stream.get('pix_fmt') == 'yuv420p'
    encoding = ['-c:v', 'copy'] if copy else ['-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p']
    inputs = input_args(video) + input_args(audio)
    subtitle_args = []
    if subtitle_path is not None:
        inputs += ['-protocol_whitelist', 'file,pipe', '-f', 'srt', '-i', subtitle_path]
        language = {'zh': 'zho', 'en': 'eng'}.get(subtitle_language, 'und')
        label = {'zh': '中文', 'en': 'English'}.get(subtitle_language, 'Translation')
        subtitle_args = ['-map', '2:s:0', '-c:s', 'mov_text',
                         '-metadata:s:s:0', f'language={language}',
                         '-metadata:s:s:0', f'handler_name={label}',
                         '-disposition:s:0', 'default']
    ffmpeg(inputs + [
        '-map', '0:v:0', '-map', '1:a:0', *subtitle_args, *encoding, '-c:a', 'aac', '-b:a', '192k',
        '-movflags', '+faststart', '-t', str(float(metadata['format']['duration'])), output])
    result = probe(output)
    if abs(float(result['format']['duration']) - float(metadata['format']['duration'])) > 0.2:
        raise RuntimeError('Output duration verification failed')

def stamp(seconds, vtt=False):
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f'{h:02}:{m:02}:{s:02}' + ('.' if vtt else ',') + f'{ms:03}'

def subtitles(segments, field, vtt=False):
    lines = ['WEBVTT', ''] if vtt else []
    for index, segment in enumerate(segments, 1):
        if segment.end <= segment.start:
            continue  # A zero-length cue cannot be rendered by subtitle players.
        text = getattr(segment, field).replace('-->', '→').replace('\r', '').replace('\n\n', '\n')
        lines.extend([str(index), f'{stamp(segment.start,vtt)} --> {stamp(segment.end,vtt)}', text, ''])
    return '\n'.join(lines) + '\n'
