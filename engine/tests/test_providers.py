import json
import httpx
import pytest
from videotranslator.config import Settings
from videotranslator.domain import Segment, ProviderError
from videotranslator.providers import DeepSeekTranslator, OpenAITranslator, create_translator, MVSeparator
from videotranslator.transcription import WhisperDiarizedTranscriber, attach_speakers
from videotranslator import media
from videotranslator.storage import LocalStorage

def transport(monkeypatch, handle):
    original=httpx.Client
    monkeypatch.setattr(httpx,'Client',lambda **kw:original(transport=httpx.MockTransport(handle),**kw))


@pytest.mark.parametrize(('code', 'name'), [('zh', 'Simplified Chinese'), ('en', 'English'),
    ('ja', 'Japanese'), ('es', 'Spanish'), ('ar', 'Arabic')])
def test_openai_translates_entire_dialogue_in_one_request(monkeypatch, code, name):
    segments = [Segment(f'seg-{i:05}', '', i, i+1, f'Line {i}') for i in range(58)]
    calls = []
    def handle(request):
        calls.append(request)
        assert request.url.path == '/v1/chat/completions'
        assert request.headers['authorization'] == 'Bearer test-only-key'
        sent = json.loads(request.content)
        assert sent['model'] == 'gpt-6-luna'
        assert sent['reasoning_effort'] == 'none'
        assert 'thinking' not in sent and 'max_tokens' not in sent
        assert sent['response_format']['json_schema']['strict'] is True
        payload = json.loads(sent['messages'][1]['content'])
        assert payload['target_language'] == name
        assert payload['terminology'] == 'Alice: keep this name'
        assert [s['id'] for s in payload['segments']] == [s.id for s in segments]
        assert payload['segments'][-1]['source_text'] == 'Line 57'
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps({'segments': [
                {'id': s.id, 'translation': f'Translation {i}'} for i, s in enumerate(segments)]})}}]})
    transport(monkeypatch, handle)
    translator = create_translator(Settings(openai_api_key='test-only-key'))
    assert isinstance(translator, OpenAITranslator)
    result = translator.translate(segments, code, 'Alice: keep this name')
    assert len(calls) == 1 and len(result) == 58
    assert result[segments[-1].id] == 'Translation 57'


@pytest.mark.parametrize(('entries', 'reason', 'refusal'), [
    ([{'id': 'seg-1', 'translation': 'hello'}], 'length', None),
    ([{'id': 'seg-1', 'translation': 'hello'}], 'stop', None),
    ([{'id': 'wrong-id', 'translation': 'hello'}], 'stop', None),
    ([{'id': 'seg-1', 'translation': 'hello'}] * 2, 'stop', None),
    ([{'id': 'seg-1', 'translation': ''}, {'id': 'seg-2', 'translation': 'hi'}], 'stop', None),
    ([], 'stop', 'refused'),
])
def test_openai_rejects_incomplete_output_without_splitting_or_fallback(monkeypatch, entries, reason, refusal):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={'choices': [{'finish_reason': reason,
            'message': {'content': json.dumps({'segments': entries}), 'refusal': refusal}}]})
    transport(monkeypatch, handle)
    with pytest.raises(ProviderError, match='no partial translation was accepted'):
        OpenAITranslator(Settings()).translate(
            [Segment('seg-1', '', 0, 1, 'Hello'), Segment('seg-2', '', 1, 2, 'Hi')], 'zh', '')
    assert len(calls) == 1


def test_translation_settings_invalidate_previous_translation_cache():
    cfg = Settings()
    assert cfg.translation_label() == 'gpt-6-luna'
    assert cfg.pipeline_config()['translation_strategy'] == 'whole-transcript-v1'
    assert cfg.pipeline_config() != Settings(translation_provider='deepseek').pipeline_config()
    assert cfg.pipeline_config() != Settings(translation_model='gpt-6-sol').pipeline_config()
    assert isinstance(create_translator(Settings(translation_provider='deepseek')), DeepSeekTranslator)
    scribe = Settings(transcription_provider='scribe', elevenlabs_api_key='test',
                      deepseek_api_key='test', openai_api_key='')
    assert scribe.missing_keys() == ['OPENAI_API_KEY']


def test_empty_transcript_does_not_call_openai(monkeypatch):
    def handle(request):
        pytest.fail('Empty transcript must not make an API request')
    transport(monkeypatch, handle)
    assert OpenAITranslator(Settings()).translate([], 'zh', '') == {}

@pytest.mark.parametrize('read_timeout', [120, 600])
def test_tokenizer_allows_cloud_model_reload_without_changing_connect_timeout(monkeypatch, read_timeout):
    from videotranslator.providers import IndexSynthesizer
    def handle(request):
        assert request.extensions['timeout']['connect'] == 120
        # A 150-second model reload exceeds the local timeout, but is valid in cloud.
        if request.extensions['timeout']['read'] < 150:
            raise httpx.ReadTimeout('model reload', request=request)
        return httpx.Response(200, json={'counts':[3]})
    transport(monkeypatch, handle)
    synth = IndexSynthesizer(Settings(tts_tokenize_timeout_seconds=read_timeout))
    if read_timeout == 120:
        with pytest.raises(httpx.ReadTimeout):
            synth.tokenize(['hello'])
    else:
        assert synth.tokenize(['hello']) == [3]


def test_synthesizer_passes_explicit_target_language(monkeypatch):
    from videotranslator.providers import IndexSynthesizer
    def handle(request):
        body = json.loads(request.content)
        assert body['language'] == 'es'
        return httpx.Response(200, json={'output': body['output']})
    transport(monkeypatch, handle)
    segment = Segment('seg-1', 'a', 0, 3, 'Hola', translation='Hola',
                      speaker_reference='jobs/test/speaker.wav', emotion_reference='jobs/test/emotion.wav')
    result = IndexSynthesizer(Settings()).synthesize(segment, 'jobs/test/output.wav', 'es')
    assert result['output'] == 'jobs/test/output.wav'


@pytest.mark.parametrize(('code', 'name'), [('ja', 'Japanese'), ('es', 'Spanish'), ('ar', 'Arabic')])
def test_translation_supports_indextts25_languages(monkeypatch, code, name):
    def handle(request):
        sent = json.loads(request.content)
        payload = json.loads(sent['messages'][1]['content'])
        assert payload['target_language'] == name
        return httpx.Response(200, json={'choices':[{'finish_reason':'stop',
            'message':{'content':json.dumps({'segments':[{'id':'seg-1','translation':'translated'}]})}}]})
    transport(monkeypatch, handle)
    result = DeepSeekTranslator(Settings()).translate([Segment('seg-1','a',0,3,'Hello')], code, '')
    assert result == {'seg-1':'translated'}

@pytest.mark.parametrize('entries,reason',[
    ([{'id':'seg-1','translation':'你好'}],'length'),
    ([{'id':'wrong-id','translation':'你好'}],'stop'),
    ([{'id':'seg-1','translation':'你好'},{'id':'seg-1','translation':'重复'}],'stop'),
])
def test_translation_rejects_partial_or_mismatched_response(monkeypatch,entries,reason):
    def handle(request):
        sent=json.loads(request.content)
        assert sent['model']=='deepseek-flash'
        assert sent['response_format']=={'type':'json_object'}
        assert sent['thinking']=={'type':'disabled'}
        return httpx.Response(200,json={'choices':[{'finish_reason':reason,'message':{'content':json.dumps({'segments':entries})}}]})
    transport(monkeypatch,handle)
    with pytest.raises(ProviderError):
        DeepSeekTranslator(Settings()).translate([Segment('seg-1','a',0,3,'Hello')],'zh','')

def test_whisper_and_independent_diarization_preserve_text_and_cache(monkeypatch,tmp_path):
    storage=LocalStorage(str(tmp_path))
    media.ffmpeg(['-f','lavfi','-i','sine=duration=6',storage.path('dialogue.wav')])
    calls=[]
    def handle(request):
        payload=request.read()
        assert request.url.path=='/v1/audio/transcriptions'
        assert request.headers['authorization']=='Bearer fixture-secret'
        if b'whisper-1' in payload:
            calls.append('whisper')
            assert b'verbose_json' in payload and b'timestamp_granularities[]' in payload
            return httpx.Response(200,json={'language':'english','text':'Hello. Goodbye.',
                'words':[{'word':'Hello.','start':0,'end':2},{'word':'Goodbye.','start':3,'end':5}]})
        calls.append('diarization')
        assert b'gpt-4o-transcribe-diarize' in payload and b'diarized_json' in payload and b'auto' in payload
        return httpx.Response(200,json={'segments':[{'start':0,'end':2,'speaker':'A','text':'Different text.'},
                                                   {'start':3,'end':5,'speaker':'B','text':'Ignored.'}]})
    transport(monkeypatch,handle)
    transcriber=WhisperDiarizedTranscriber(Settings(openai_api_key='fixture-secret'),storage)
    result=transcriber.transcribe('dialogue.wav')
    encoded=media.probe(storage.path('recognition/dialogue.mp3'))
    stream=encoded['streams'][0]
    assert stream['codec_name']=='mp3'
    assert int(stream['sample_rate'])==16000 and stream['channels']==1
    assert int(stream['bit_rate'])==48000
    assert abs(float(encoded['format']['duration'])-6)<.2
    assert int(media.probe(storage.path('dialogue.wav'))['streams'][0]['sample_rate'])==44100
    assert [w['speaker_id'] for w in result['words']]==['A','B']
    assert [w['text'] for w in result['words']]==['Hello.','Goodbye.']
    assert transcriber.transcribe('dialogue.wav')==result
    assert calls==['whisper','diarization']

def test_ambiguous_speakers_are_not_collapsed_into_one_voice():
    result=attach_speakers({'text':'Hello','words':[{'word':'Hello','start':0,'end':2}]},
        {'segments':[{'start':0,'end':2,'speaker':'A'},{'start':1,'end':2,'speaker':'B'}]},3)
    assert result['words'][0]['speaker_id']=='unknown'
    assert result['speaker_assignment']['unresolved_words']==1

def test_whisper_punctuation_and_small_independent_boundary_drift():
    result=attach_speakers({'text':'Hello! Welcome to the studio!', 'words':[
        {'word':'Hello','start':0,'end':.76},{'word':'Welcome','start':2.1,'end':2.2},
        {'word':'to','start':2.2,'end':2.56},{'word':'the','start':2.56,'end':2.98},
        {'word':'studio','start':4.42,'end':5.06}]},
        {'segments':[{'start':0,'end':.6,'speaker':'A'},{'start':.65,'end':4.65,'speaker':'A'}]},6)
    assert result['words'][0]['text']=='Hello!'
    assert result['words'][-1]['text']=='studio!'
    assert all(w['speaker_id']=='A' for w in result['words'])
    assert result['speaker_assignment']['boundary_tolerance_words']==1

def test_boundary_tolerance_does_not_hide_a_nearby_second_speaker():
    result=attach_speakers({'text':'Hello','words':[{'word':'Hello','start':4.42,'end':5.06}]},
        {'segments':[{'start':0,'end':4.65,'speaker':'A'},{'start':5.1,'end':6,'speaker':'B'}]},6)
    assert result['words'][0]['speaker_id']=='unknown'

def test_point_words_preserve_text_and_timestamps_in_utterance():
    from videotranslator.segmentation import parse_words, segment_dialogue
    result=attach_speakers({'text':'Dr. J, hello.', 'words':[
        {'word':'Dr','start':0,'end':.4}, {'word':'J','start':.66,'end':.66},
        {'word':'hello','start':1,'end':2}]},
        {'segments':[{'start':0,'end':2,'speaker':'A'}]},3)
    assert result['words'][1]['start']==result['words'][1]['end']==.66
    segments=segment_dialogue(parse_words(result))
    # Stable v5 uses a 300 ms pause boundary; a speaker label must not merge it.
    assert [s.source_text for s in segments] == ['Dr. J,', 'hello.']
    assert [(s.start,s.end) for s in segments] == [(0,.66),(1,2)]
    assert result['speaker_assignment']['zero_duration_words']==1

def test_point_word_at_speaker_change_is_ambiguous_and_requires_review():
    from videotranslator.segmentation import parse_words, segment_dialogue
    result=attach_speakers({'text':'Hi.', 'words':[{'word':'Hi','start':1,'end':1}]},
        {'segments':[{'start':0,'end':1,'speaker':'A'}, {'start':1,'end':2,'speaker':'B'}]},3)
    segment=segment_dialogue(parse_words(result))[0]
    assert result['words'][0]['speaker_id']=='unknown'
    assert segment.speaker_id==''  # Boundary-only pipeline has no identity bank.
    assert 'zero_duration_utterance' in segment.flags
    assert segment.source_text=='Hi.'

@pytest.mark.parametrize('end',[float('nan'),float('inf'),20])
def test_whisper_invalid_timestamps_fail(end):
    with pytest.raises(ProviderError):
        attach_speakers({'text':'Hello','words':[{'word':'Hello','start':0,'end':end}]},
            {'segments':[{'start':0,'end':2,'speaker':'A'}]},3)

def test_mvsep_resume_uses_existing_job_without_upload(monkeypatch,tmp_path):
    storage=LocalStorage(str(tmp_path))
    def handle(request):
        assert request.method=='GET'  # no duplicate paid create
        if request.url.path.endswith('/separation/get'):
            assert request.url.params['hash']=='already-submitted'
            return httpx.Response(200,json={'status':'done','data':{'files':[
                {'type':kind,'url':f'https://de2.mvsep.com/{kind}.wav'} for kind in ['speech','music','sfx']]}})
        return httpx.Response(200,content=b'fixture-stem')
    transport(monkeypatch,handle)
    result=MVSeparator(Settings(),storage).separate('unused.flac','run',lambda _:None,
        {'hash':'already-submitted'},lambda:None)
    assert all(storage.exists(k) for k in [result.dialogue,result.music,result.effects])
