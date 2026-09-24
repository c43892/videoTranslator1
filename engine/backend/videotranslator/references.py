"""Both IndexTTS prompts always come from the current original utterance."""
import math
import numpy as np
import soundfile as sf


def segment_reference(segment, storage, prefix, max_seconds=15):
    if not math.isfinite(max_seconds) or not 1 <= max_seconds <= 15:
        raise ValueError('Reference conditioning window must be between 1 and 15 seconds')
    source = storage.path(segment.original_audio)
    info = sf.info(source)
    if 1 <= info.duration <= max_seconds:
        return segment.original_audio
    frames = min(info.frames, round(max_seconds * info.samplerate))
    start = 0
    if info.duration > max_seconds:
        # Select a contiguous excerpt from THIS utterance; keep its full text
        # and timeline. The energy envelope uses bounded memory.
        step = max(1, round(info.samplerate / 4))
        energy = []
        with sf.SoundFile(source) as audio:
            while True:
                block = audio.read(step, dtype='float32', always_2d=True)
                if not len(block):
                    break
                energy.append(float(np.sum(np.minimum(block.astype('float64') ** 2, .25))))
        width = max(1, math.ceil(frames / step))
        start = min(int(np.argmax(np.convolve(energy, np.ones(width), mode='valid'))) * step, info.frames - frames)
        segment.notes.append(f'current_segment_reference_excerpt:{start/info.samplerate:.3f}-{(start+frames)/info.samplerate:.3f}s')
    key = f'{prefix}/references/{segment.id}-{frames}f-{start}f.wav'
    with sf.SoundFile(source) as audio:
        audio.seek(start)
        signal = audio.read(frames, dtype='float32', always_2d=True)
    if len(signal) < info.samplerate:
        # Pad context for feature extraction, without borrowing another voice.
        signal = np.pad(signal, ((0, info.samplerate-len(signal)), (0, 0)))
        segment.notes.append('current_segment_reference_silence_padded')
    destination = storage.path(key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, signal, info.samplerate, subtype='PCM_16')
    return key
