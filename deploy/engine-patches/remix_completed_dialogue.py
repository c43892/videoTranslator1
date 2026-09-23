"""Run inside the patched engine to repair a completed mix without provider calls.

Writes a separate correction artifact and evidence; never changes billing, job
state, the source manifest, or previous results. Publish only after validation.
"""
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from videotranslator import media
from videotranslator.domain import Segment
from videotranslator.storage import LocalStorage


def remix(job_id):
    job_id = str(uuid.UUID(job_id))
    storage = LocalStorage('/data')
    manifest = storage.read_json(f'jobs/{job_id}/manifest.json')
    if not manifest.get('completed', {}).get('assemble'):
        raise ValueError('Only completed jobs can be remixed')
    segments = [Segment(**item) for item in manifest['segments']]
    if any(s.render_status not in ('translated', 'original') for s in segments):
        raise ValueError('Unfinished segment')
    prefix = f'jobs/{job_id}/runs/{manifest["fingerprint"]}'
    destination = storage.path(f'jobs/{job_id}/corrections/dialogue-v2')
    destination.mkdir(parents=True, exist_ok=True)
    dialogue, mixed, video = [destination / name for name in ('dialogue.wav','mixed.wav','translated.mp4')]
    duration = manifest['duration']
    media.dialogue_timeline(segments, storage, dialogue, duration,
                           original=storage.path(f'{prefix}/stems/dialogue.wav'))
    media.mix(dialogue, storage.path(f'{prefix}/stems/music.wav'),
              storage.path(f'{prefix}/stems/effects.wav'), mixed, duration)
    source = storage.path(f'jobs/{job_id}/input.mp4')
    media.assemble(source, mixed, video, media.probe(source))
    media.ffmpeg(['-xerror', '-i', video, '-f', 'null', '-'])
    # Measure uncovered intervals in bounded blocks, not a full-film allocation.
    sum_old = sum_new = samples = 0
    # The previous mix policy copied this source stem into every uncovered gap.
    with sf.SoundFile(storage.path(f'{prefix}/stems/dialogue.wav')) as old, sf.SoundFile(dialogue) as new:
        offset = 0
        while True:
            a, b = old.read(48000, always_2d=True), new.read(48000, always_2d=True)
            if not len(a):
                break
            a, b = a[:len(b)], b[:len(a)]
            mask = np.ones(len(a), dtype=bool)
            for s in segments:
                lo, hi = max(0,round(s.start*48000)-offset), min(len(a),round(s.end*48000)-offset)
                if hi > lo:
                    mask[lo:hi] = False
            sum_old += float(np.sum(a[mask] ** 2))
            sum_new += float(np.sum(b[mask] ** 2))
            samples += int(mask.sum()) * 2
            offset += len(a)
    evidence = {'job_id':job_id, 'policy':'translated_and_explicit_fallback_only',
        'uncovered_seconds':samples/96000, 'source_uncovered_rms':(sum_old/max(1,samples))**.5,
        'new_uncovered_rms':(sum_new/max(1,samples))**.5,
        'duration':duration, 'video':str(video), 'audio':str(mixed),
        'source_manifest':f'jobs/{job_id}/manifest.json', 'provider_calls':0}
    (destination/'evidence.json').write_text(json.dumps(evidence,indent=2))
    return evidence


if __name__ == '__main__':
    print(json.dumps(remix(sys.argv[1])))
