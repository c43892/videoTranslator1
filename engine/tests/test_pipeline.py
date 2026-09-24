import shutil
from pathlib import PurePosixPath
from types import SimpleNamespace
import numpy as np
import soundfile as sf
import pytest
from videotranslator import media
from videotranslator.config import Settings
from videotranslator.domain import Stems, Cancelled
from videotranslator.pipeline import Pipeline
from videotranslator.storage import LocalStorage

class Separator:
    calls = 0
    def __init__(self, storage):
        self.storage = storage
    def separate(self, audio, prefix, checkpoint, remote, check_cancel):
        self.calls += 1
        checkpoint({'hash':'fixture-remote'})
        keys = {}
        for name in ('dialogue','music','effects'):
            key=f'{prefix}/{name}.wav'
            data = np.sin(np.arange(8*48000)*2*np.pi*220/48000)*(.1 if name=='dialogue' else .01)
            sf.write(self.storage.path(key),data,48000)
            keys[name]=key
        return Stems(**keys)
class Transcriber:
    calls=0
    def transcribe(self,audio):
        self.calls+=1
        return {'language_code':'en','text':'Hello world. Another sentence.',
                'words':[{'text':'Hello','start':.2,'end':1,'speaker_id':'a','type':'word'},
                         {'text':'world.','start':1,'end':3.4,'speaker_id':'a','type':'word'},
                         {'text':'Another','start':4,'end':5,'speaker_id':'a','type':'word'},
                         {'text':'sentence.','start':5,'end':7.6,'speaker_id':'a','type':'word'}]}
class Translator:
    calls=0
    def translate(self,segments,language,terminology):
        self.calls+=1
        return {s.id:'你好世界。' for s in segments}
class Synthesizer:
    def __init__(self,storage):
        self.storage=storage
        self.calls=[]
    def tokenize(self,texts):
        return [5]*len(texts)
    def synthesize(self,segment,output):
        self.calls.append((segment.speaker_reference,segment.emotion_reference))
        self.storage.path(output).parent.mkdir(parents=True,exist_ok=True)
        sf.write(self.storage.path(output),np.sin(np.arange(48000)*2*np.pi*330/48000)*.1,48000)
        return {'duration':1}

def test_real_media_end_to_end_with_provider_fixtures_and_resume(tmp_path):
    storage=LocalStorage(str(tmp_path))
    video=storage.path('input.mp4')
    media.ffmpeg(['-f','lavfi','-i','color=c=blue:s=160x90:r=24:d=8',
                  '-f','lavfi','-i','sine=frequency=220:duration=8','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',video])
    config=Settings(storage_root=str(tmp_path))
    providers=Separator(storage),Transcriber(),Translator(),Synthesizer(storage)
    pipeline=Pipeline(config,storage,*providers)
    job=SimpleNamespace(id='test-job',input_key='input.mp4',target_language='zh',terminology='')
    reports=[]
    output=pipeline.run(job,lambda s,p:reports.append((s,p)),lambda:None)
    assert storage.exists(output['video'])
    assert abs(float(media.probe(storage.path(output['video']))['format']['duration'])-8)<.2
    manifest=storage.read_json(output['manifest'])
    assert len(manifest['segments'])==2
    refs=providers[3].calls
    assert refs[0][0]!=refs[1][0]  # each utterance supplies its own voice
    assert all(voice==emotion for voice,emotion in refs)
    assert refs[0][1]!=refs[1][1]  # emotion belongs to each utterance
    assert reports[-1]==('complete',100)
    pipeline.run(job,lambda *_:None,lambda:None)
    assert providers[0].calls==1 and providers[1].calls==1 and providers[2].calls==1
    assert len(providers[3].calls)==2
    storage.write_json('jobs/test-job/overrides.json',{'translations':{'seg-00001':'修改后的译文。'},'speaker_references':{}})
    pipeline.run(job,lambda *_:None,lambda:None)
    assert len(providers[3].calls)==3

def test_cancel_before_any_provider_call(tmp_path):
    config=Settings(storage_root=str(tmp_path))
    storage=LocalStorage(str(tmp_path))
    providers=Separator(storage),Transcriber(),Translator(),Synthesizer(storage)
    pipeline=Pipeline(config,storage,*providers)
    def stop():
        raise Cancelled()
    with pytest.raises(Cancelled):
        pipeline.run(SimpleNamespace(id='cancelled',target_language='zh',terminology='',input_key='input.mp4'),lambda *_:None,stop)
    assert providers[0].calls==0

@pytest.mark.parametrize('failure',['overflow','incomplete','text_limit','reference'])
def test_bad_segment_falls_back_continues_and_can_be_retried_alone(tmp_path,failure):
    from videotranslator.domain import NeedsReview
    storage=LocalStorage(str(tmp_path))
    media.ffmpeg(['-f','lavfi','-i','color=s=160x90:r=24:d=8',
                  '-f','lavfi','-i','sine=frequency=220:duration=8',
                  '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',storage.path('input.mp4')])
    class FailingSynth(Synthesizer):
        failing=True
        def tokenize(self,texts):
            return [999 if self.failing and failure=='text_limit' and i==0 else 5 for i,_ in enumerate(texts)]
        def synthesize(self,segment,output):
            if self.failing and segment.id=='seg-00001':
                if failure=='incomplete':
                    raise NeedsReview('Incomplete synthesis')
                if failure=='overflow':
                    self.calls.append((segment.speaker_reference,segment.emotion_reference))
                    self.storage.path(output).parent.mkdir(parents=True,exist_ok=True)
                    sf.write(self.storage.path(output),np.ones(5*48000)*.1,48000)
                    return
            super().synthesize(segment,output)
    class ReferenceTranscriber(Transcriber):
        def transcribe(self,audio):
            result=super().transcribe(audio)
            if failure=='reference':
                for word in result['words'][:2]: word['speaker_id']='unknown'
            return result
    providers=Separator(storage),ReferenceTranscriber(),Translator(),FailingSynth(storage)
    pipeline=Pipeline(Settings(storage_root=str(tmp_path)),storage,*providers)
    job=SimpleNamespace(id='fallback',input_key='input.mp4',target_language='zh',terminology='')
    output=pipeline.run(job,lambda *_:None,lambda:None)
    manifest=storage.read_json(output['manifest'])
    first,second=manifest['segments']
    if failure in ('overflow', 'reference', 'text_limit'):
        assert all(s['render_status'] == 'translated' for s in manifest['segments'])
        assert not manifest['warnings']
        return  # Best-effort references, native text chunks and forced duration fitting.
    assert first['render_status']=='original' and first['flags']
    assert second['render_status']=='translated' and not second['flags']
    assert len(manifest['warnings'])==1 and 'seg-00001' in manifest['warnings'][0]
    assert storage.exists(output['video'])
    assert abs(float(media.probe(storage.path(output['video']))['format']['duration'])-8)<.1
    timeline=storage.path(str(PurePosixPath(output['audio']).parent/'dialogue-translated.wav'))
    rendered,sr=sf.read(timeline)
    original,_=sf.read(storage.path(manifest['stems']['dialogue']))
    assert np.array_equal(rendered[int(.2*sr):int(3.4*sr)],original[int(.2*sr):int(3.4*sr)])
    if failure in ('reference','text_limit'):
        return
    before=len(providers[3].calls)
    providers[3].failing=False
    storage.write_json('jobs/fallback/overrides.json',{'synthesis_revisions':{'seg-00001':'retry-1'}})
    pipeline.run(job,lambda *_:None,lambda:None)
    assert len(providers[3].calls)==before+1  # second segment reused
    assert providers[0].calls==providers[1].calls==providers[2].calls==1
    manifest=storage.read_json(output['manifest'])
    assert not manifest['warnings']
    assert all(s['render_status']=='translated' for s in manifest['segments'])
