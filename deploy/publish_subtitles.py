"""Add switchable subtitles to a completed Studio video without resynthesis or billing."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from videotranslator.adapters.docker_engine import DockerEngineBackend, validate_output
from videotranslator.adapters.ffmpeg import ffprobe_json
from videotranslator.adapters.local_storage import LocalObjectStorage
from videotranslator.config import settings_from_env
from videotranslator.docstore import SQLiteStore
from videotranslator.domain.models import Job


def publish(job_id):
    settings = settings_from_env()
    if settings.profile != 'local-full' or settings.engine_backend != 'docker':
        raise ValueError('Trusted local engine required')
    store = SQLiteStore(settings.store_path)
    storage = LocalObjectStorage(settings.local_storage_dir)
    with store.transaction() as tx:
        job = tx.get(Job, job_id)
        if not job or job.status != 'succeeded' or job.media_type != 'video' or job.assets_deleted_at:
            raise ValueError('Completed video task required')
    if job.output_object_key.endswith('-subtitles-v1.mp4'):
        print(json.dumps({'already_published': True})); return
    backend = DockerEngineBackend(storage, docker=settings.docker_command, container=settings.engine_container)
    data = backend._call('status', job.backend_job_id)
    key = str(Path(job.output_object_key).with_name(Path(job.output_object_key).stem+'-subtitles-v1.mp4'))
    source = storage.local_path(job.output_object_key)
    with tempfile.TemporaryDirectory(prefix='vt-subtitles-') as directory:
        root = Path(directory)
        for extension in ('srt', 'vtt'):
            path = data['outputs']['translated_'+extension]
            if not path.startswith('/data/jobs/'+data['id']+'/'):
                raise ValueError('Unexpected subtitle location')
            backend._command(['cp', f'{settings.engine_container}:{path}', str(root/('translated.'+extension))])
        output = root/'result.mp4'
        language = {'zh':'zho', 'en':'eng'}.get(job.target_language, 'und')
        label = {'zh':'中文', 'en':'English'}.get(job.target_language, 'Translation')
        subprocess.run(['ffmpeg','-v','error','-i',str(source),'-i',str(root/'translated.srt'),
            '-map','0:v:0','-map','0:a:0','-map','1:s:0','-c:v','copy','-c:a','copy','-c:s','mov_text',
            '-metadata:s:s:0','language='+language,'-metadata:s:s:0','handler_name='+label,
            '-disposition:s:0','default','-movflags','+faststart',str(output)], check=True)
        validate_output(output, job.duration_ms)
        streams = ffprobe_json(output)['streams']
        assert any(s['codec_name'] == 'mov_text' for s in streams)
        def av_hash(path):
            return subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map','0:v:0',
                '-map','0:a:0','-c','copy','-f','streamhash','-'])
        if av_hash(source) != av_hash(output):
            raise ValueError('Existing translated video/audio changed')
        evidence = {'studio_job_id':job_id, 'previous_output_key':job.output_object_key,
                    'new_output_key':key, 'audio_video_unchanged':True, 'subtitle_language':language,
                    'additional_charge_cents':0}
        evidence_path = root/'evidence.json'
        evidence_path.write_text(json.dumps(evidence, indent=2))
        storage.upload(root/'translated.vtt', key+'.vtt')
        storage.upload(evidence_path, key+'.evidence.json')
        storage.upload(output, key)
    with store.transaction() as tx:
        current = tx.get(Job, job_id)
        if current.status != 'succeeded' or current.output_object_key != job.output_object_key or current.assets_deleted_at:
            raise ValueError('Task changed during subtitle publication')
        current.output_object_key = key
        tx.put(current, job_id)
    print(json.dumps(evidence))


if __name__ == '__main__':
    publish(sys.argv[1])
