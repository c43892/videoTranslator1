import json
from types import SimpleNamespace
import numpy as np
import soundfile as sf
import httpx
import pytest
from videotranslator import media
from videotranslator.domain import Word, Segment, ProviderError
from videotranslator.config import Settings
from videotranslator.scribe import normalize_scribe, ScribeTranscriber, create_transcriber
from videotranslator.segmentation import segment_dialogue, parse_words
from videotranslator.storage import LocalStorage


def response():
    return {'language_code':'jpn','text':'はい (laughter) そうです。','words':[
        {'type':'word','text':'はい','start':0,'end':1,'speaker_id':'ignored'},
        {'type':'spacing','text':' '},
        {'type':'audio_event','text':'(laughter)','start':1,'end':2},
        {'type':'word','text':'そうです。','start':2,'end':4}]}


def test_events_are_preserved_but_never_translated_or_assigned_identity():
    data=normalize_scribe(response(),4)
    assert data['text']=='はいそうです。'
    assert data['audio_events']==[{'type':'audio_event','text':'(laughter)','start':1.,'end':2.}]
    assert all(w['speaker_id']=='' for w in data['words'])
    segs=segment_dialogue(parse_words(data),audio_events=data['audio_events'])
    assert len(segs)==2
    assert [(s.start,s.end) for s in segs]==[(0,1),(2,4)]
    assert all(not s.flags for s in segs)


def test_short_merge_can_use_same_side_of_event_only():
    words=[Word('First,',0,.5,''),Word('second.',.5,1,''),Word('After',2,3,''),Word('that.',3,6,'')]
    segs=segment_dialogue(words,audio_events=[{'start':1,'end':2}])
    assert len(segs)==2 and [(s.start,s.end) for s in segs]==[(0,1),(2,6)]


@pytest.mark.parametrize('bad',[{'words':None},{'words':[{'type':'unexpected','text':'hi','start':0,'end':1}]},
    {'words':[{'type':'audio_event','text':'laugh','start':-1,'end':2}]}])
def test_invalid_transcripts_are_not_cached(bad):
    with pytest.raises(ProviderError):normalize_scribe(bad,4)


def test_scribe_request_cache_and_provider_selection(tmp_path,monkeypatch):
    store=LocalStorage(str(tmp_path));store.path('recognition').mkdir();store.path('recognition/scribe-input.flac').write_bytes(b'fixture')
    monkeypatch.setattr(media,'probe',lambda _: {'format':{'duration':'4'}})
    requests=[];real=httpx.Client
    def handle(request):
        requests.append(request)
        assert request.headers['xi-api-key']=='test-only-key'
        body=request.content.decode()
        for field,value in [('model_id','scribe_v2' if len(requests)==1 else 'scribe_v1'),('timestamps_granularity','word'),('diarize','false'),('tag_audio_events','true')]:
            assert f'name="{field}"\r\n\r\n{value}' in body
        return httpx.Response(200,json=response())
    monkeypatch.setattr(httpx,'Client',lambda **kw: real(transport=httpx.MockTransport(handle),**kw))
    config=Settings(transcription_provider='scribe',elevenlabs_api_key='test-only-key',deepseek_api_key='test',openai_api_key='')
    assert 'OPENAI_API_KEY' not in config.missing_keys()
    adapter=create_transcriber(config,store)
    assert isinstance(adapter,ScribeTranscriber)
    assert adapter.transcribe('source.wav')==adapter.transcribe('source.wav')
    assert len(requests)==1
    config.scribe_model='scribe_v1';create_transcriber(config,store).transcribe('source.wav')
    assert len(requests)==2


def test_error_does_not_expose_provider_body(tmp_path,monkeypatch):
    store=LocalStorage(str(tmp_path));store.path('recognition').mkdir();store.path('recognition/scribe-input.flac').write_bytes(b'fixture')
    monkeypatch.setattr(media,'probe',lambda _: {'format':{'duration':'4'}})
    real=httpx.Client
    monkeypatch.setattr(httpx,'Client',lambda **kw: real(transport=httpx.MockTransport(lambda _:httpx.Response(401,text='secret-body')),**kw))
    with pytest.raises(ProviderError,match='authentication') as error:
        ScribeTranscriber(Settings(elevenlabs_api_key='test'),store).transcribe('source.wav')
    assert 'secret-body' not in str(error.value)


def test_service_status_reports_selected_recognizer(monkeypatch):
    from videotranslator import api
    monkeypatch.setattr(api, 'cfg', Settings(transcription_provider='scribe'))
    monkeypatch.setattr(api.httpx, 'get', lambda *a, **kw: httpx.Response(200, json={'status':'ready'}))
    providers = api.services(user=object())['providers']
    assert providers['transcription'] == 'scribe_v2'
    assert providers['diarization'] == 'disabled'


def test_mixer_preserves_unrecognized_sounds_and_replaces_only_successes(tmp_path):
    store=LocalStorage(str(tmp_path));sr=48000
    original=np.full((6*sr,2),.1,dtype=np.float32)
    original[sr:2*sr]=.2  # laughter
    sf.write(store.path('source.wav'),original,sr,subtype='FLOAT')
    sf.write(store.path('dub.wav'),np.full((sr,2),.3),sr,subtype='FLOAT')
    segs=[Segment('good','',2,3,'speech',aligned_audio='dub.wav',render_status='translated'),
          Segment('bad','',4,5,'speech',render_status='original')]
    media.dialogue_timeline(segs,store,store.path('result.wav'),6,original=store.path('source.wav'),preserve_intervals=[{'start':1,'end':2}])
    out,_=sf.read(store.path('result.wav'))
    assert np.array_equal(out[:2*sr],original[:2*sr])
    assert np.allclose(out[2*sr:3*sr],.3)
    assert np.array_equal(out[3*sr:],original[3*sr:])


def test_overlapping_events_are_preserved_without_doubling_source(tmp_path):
    store=LocalStorage(str(tmp_path));sr=48000
    sf.write(store.path('source.wav'),np.full((3*sr,2),.1),sr,subtype='FLOAT')
    sf.write(store.path('dub.wav'),np.full((sr,2),.3),sr,subtype='FLOAT')
    segs=[Segment('a','',1,2,'speech',aligned_audio='dub.wav',render_status='translated')]
    media.dialogue_timeline(segs,store,store.path('out.wav'),3,original=store.path('source.wav'),
        preserve_intervals=[{'start':1.2,'end':1.5},{'start':1.4,'end':1.6}])
    out,_=sf.read(store.path('out.wav'))
    assert np.allclose(out[round(1.2*sr):round(1.6*sr)],.4)
    assert np.allclose(out[sr:round(1.2*sr)],.3)


def test_pipeline_carries_events_into_segmentation_and_mix(tmp_path):
    import sys
    sys.path.insert(0,'/app/tests')
    from test_pipeline import Separator, Translator, Synthesizer, Pipeline
    store=LocalStorage(str(tmp_path))
    media.ffmpeg(['-f','lavfi','-i','color=s=160x90:r=24:d=8','-f','lavfi','-i','sine=duration=8',
        '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',store.path('input.mp4')])
    class Events:
        def transcribe(self,audio):return normalize_scribe(response(),8)
    output=Pipeline(Settings(),store,Separator(store),Events(),Translator(),Synthesizer(store)).run(
        SimpleNamespace(id='events',input_key='input.mp4',target_language='zh',terminology=''),lambda *_:None,lambda:None)
    manifest=store.read_json(output['manifest']);assert len(manifest['segments'])==2
    assert manifest['audio_events'][0]['text']=='(laughter)'
    assert all('(laughter)' not in s['source_text'] for s in manifest['segments'])
    from pathlib import PurePosixPath
    timeline=store.path(str(PurePosixPath(output['audio']).parent/'dialogue-translated.wav'))
    actual,_=sf.read(timeline);original,_=sf.read(store.path(manifest['stems']['dialogue']))
    assert np.array_equal(actual[48000:96000],original[48000:96000])
