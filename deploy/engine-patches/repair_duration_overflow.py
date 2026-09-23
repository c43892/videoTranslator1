"""Re-align cached successful synthesis and remix a completed task, without model calls."""
import copy
import json
import sys
import uuid

import soundfile as sf
from videotranslator import media
from videotranslator.domain import Segment
from videotranslator.storage import LocalStorage


def repair(job_id):
    job_id = str(uuid.UUID(job_id))
    storage = LocalStorage('/data')
    key = f'jobs/{job_id}/manifest.json'
    original = storage.read_json(key)
    if not original.get('completed',{}).get('assemble'):
        raise ValueError('A completed task is required')
    manifest = copy.deepcopy(original)
    prefix = f'jobs/{job_id}/corrections/tempo-v1'
    destination = storage.path(prefix); destination.mkdir(parents=True,exist_ok=True)
    segments = [Segment(**item) for item in manifest['segments']]
    repaired = []
    for segment in segments:
        if segment.render_status != 'original' or not segment.flags:
            continue
        if not all(flag.startswith('duration_overflow:') for flag in segment.flags):
            continue
        if not segment.translation or not segment.synthesized_audio or not storage.exists(segment.synthesized_audio):
            continue
        generated = sf.info(storage.path(segment.synthesized_audio)).duration
        window = segment.end-segment.start
        segment.aligned_audio = f'{prefix}/{segment.id}.wav'
        media.align(storage.path(segment.synthesized_audio),storage.path(segment.aligned_audio),window,1.25)
        actual = sf.info(storage.path(segment.aligned_audio)).duration
        if abs(actual-window) > 1/48000:
            raise ValueError('Aligned duration mismatch')
        repaired.append({'segment':segment.id,'generated_seconds':generated,'window_seconds':window,
                         'aligned_seconds':actual,'tempo':generated/window})
        segment.flags = []
        segment.render_status = 'translated'
        segment.notes.append(f'pitch_preserving_tempo:{generated/window:.6f}')
    if not repaired:
        raise ValueError('No duration-only fallback with cached synthesis was found')
    stems = manifest['stems']
    dialogue,mixed,video = [storage.path(f'{prefix}/{name}') for name in ('dialogue.wav','mixed.wav','translated.mp4')]
    media.dialogue_timeline(segments,storage,dialogue,manifest['duration'],original=storage.path(stems['dialogue']))
    media.mix(dialogue,storage.path(stems['music']),storage.path(stems['effects']),mixed,manifest['duration'])
    source = storage.path(f'jobs/{job_id}/input.mp4')
    media.assemble(source,mixed,video,media.probe(source))
    media.ffmpeg(['-xerror','-i',video,'-f','null','-'])
    ids = {item['segment'] for item in repaired}
    manifest['warnings'] = [w for w in manifest.get('warnings',[]) if not (
        'duration_overflow:' in w and any(w.startswith(sid+'（') for sid in ids))]
    manifest['segments'] = [s.to_dict() for s in segments]
    manifest['alignment_policy'] = 'pitch-preserving-force-fit-v1'
    manifest['outputs'].update(video=f'{prefix}/translated.mp4',audio=f'{prefix}/mixed.wav',manifest=f'{prefix}/manifest.json')
    storage.write_json(f'{prefix}/manifest.json',manifest)
    storage.write_json(f'{prefix}/original-manifest.json',original)
    evidence = {'job_id':job_id,'repaired':repaired,'provider_calls':0,'warnings':manifest['warnings'],
                'outputs':manifest['outputs'],'previous_outputs':original['outputs']}
    storage.write_json(f'{prefix}/evidence.json',evidence)
    return evidence


def activate(job_id):
    from videotranslator.db import Session, Job
    job_id = str(uuid.UUID(job_id))
    storage = LocalStorage('/data')
    prefix = f'jobs/{job_id}/corrections/tempo-v1'
    manifest = storage.read_json(f'{prefix}/manifest.json')
    evidence = storage.read_json(f'{prefix}/evidence.json')
    if evidence['job_id'] != job_id or not evidence['repaired']:
        raise ValueError('Correction mismatch')
    with Session.begin() as db:
        job = db.get(Job,job_id)
        if not job or job.status not in ('completed','completed_with_warnings'):
            raise ValueError('Task is not completed')
        if job.outputs not in (evidence['previous_outputs'],manifest['outputs']):
            raise ValueError('Task output changed since correction')
        # The exact previous manifest and output files remain in the correction evidence.
        storage.write_json(f'jobs/{job_id}/manifest.json',manifest)
        job.outputs = manifest['outputs']
        job.status = 'completed_with_warnings' if manifest['warnings'] else 'completed'
    return {'job_id':job_id,'activated':True}


if __name__ == '__main__':
    action = activate if '--activate' in sys.argv else repair
    print(json.dumps(action(sys.argv[1]),ensure_ascii=False))
