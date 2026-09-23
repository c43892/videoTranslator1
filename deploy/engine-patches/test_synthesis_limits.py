import sys
import json
import numpy as np
import soundfile as sf
import httpx
import pytest
from videotranslator.config import Settings
from videotranslator.domain import Segment, Word, ProviderError
from videotranslator.storage import LocalStorage
from videotranslator.references import segment_reference
from videotranslator.punctuation import (valid_boundary_suggestions, indexed_punctuation,
                                        apply_punctuation, DeepSeekPunctuator)


def test_alignment_uses_exact_output_frames_for_whisper_timestamps(tmp_path):
    from videotranslator import media
    sf.write(tmp_path/'tts.wav',np.sin(np.arange(7*22050)*2*np.pi*220/22050)*.2,22050)
    window=5.10003662109375
    media.align(tmp_path/'tts.wav',tmp_path/'aligned.wav',window,1.25)
    info=sf.info(tmp_path/'aligned.wav')
    assert info.samplerate==48000 and info.frames==round(window*48000)


def test_bounded_reference_uses_current_segment_without_cutting_utterance(tmp_path):
    storage = LocalStorage(str(tmp_path))
    rate = 16000
    signal = np.concatenate([np.zeros(12*rate), np.sin(np.arange(15*rate)*2*np.pi*220/rate)*.2])
    sf.write(storage.path('original.wav'), signal, rate)
    segment = Segment('s', 'A', 280, 307, 'Complete source', translation='Complete translation', original_audio='original.wav')
    key = segment_reference(segment, storage, 'test')
    clipped, _ = sf.read(storage.path(key))
    original, _ = sf.read(storage.path('original.wav'))
    assert len(clipped) == 15*rate
    assert np.array_equal(clipped, original[12*rate:])
    assert segment.original_audio == 'original.wav'
    assert (segment.start, segment.end, segment.source_text, segment.translation) == (280,307,'Complete source','Complete translation')


def test_short_reference_pads_only_its_own_audio(tmp_path):
    storage=LocalStorage(str(tmp_path))
    data=np.ones(2240)*.1
    sf.write(storage.path('short.wav'),data,16000)
    segment=Segment('short','',1,1.14,'Hi',original_audio='short.wav')
    key=segment_reference(segment,storage,'test')
    result,_=sf.read(storage.path(key));original,_=sf.read(storage.path('short.wav'))
    assert len(result)==16000 and np.array_equal(result[:2240],original)
    assert np.all(result[2240:]==0)


def test_whisper_does_not_request_speaker_labels(tmp_path,monkeypatch):
    from videotranslator.transcription import WhisperTranscriber
    from videotranslator import media
    storage=LocalStorage(str(tmp_path))
    storage.path('recognition').mkdir()
    storage.path('recognition/dialogue.mp3').write_bytes(b'cached')
    storage.write_json('recognition/whisper.json',{'text':'Hello world.','language':'en','words':[
        {'word':'Hello','start':0,'end':1},{'word':'world','start':1,'end':4}]})
    monkeypatch.setattr(media,'probe',lambda path:{'format':{'duration':'5'}})
    transcriber=WhisperTranscriber(Settings(),storage)
    transcriber.prepare_audio=lambda *args:storage.path('recognition/dialogue.mp3')
    result=transcriber.transcribe('dialogue.wav')
    assert all(w['speaker_id']=='' for w in result['words'])
    assert result['raw_responses']=={'whisper':'recognition/whisper.json'}
    assert not storage.exists('recognition/speakers.json')


def test_partial_punctuation_retains_valid_marks_and_exact_text():
    words = [Word('私',0,1,'A'),Word('です',1,4,'A'),Word('次',4,5,'A')]
    valid, rejected = valid_boundary_suggestions(words,[{'after':1,'mark':'が'}, {'after':2,'mark':'。'}])
    result = apply_punctuation(words,indexed_punctuation(words,valid))
    assert rejected == 1 and [w.text for w in result] == ['私','です。','次']
    assert [(w.start,w.end,w.speaker_id) for w in result] == [(w.start,w.end,w.speaker_id) for w in words]


def test_conflicting_suggestions_are_not_guessed():
    words = [Word('Hello',0,1,'A'),Word('again',1,4,'A')]
    valid,rejected = valid_boundary_suggestions(words,[{'after':1,'mark':'.'}, {'after':1,'mark':','}, {'after':2,'mark':'.'}])
    assert valid == [{'after':2,'mark':'.'}] and rejected == 2
    with pytest.raises(ProviderError):
        valid_boundary_suggestions(words,[{'after':1,'mark':'の'}])


def test_partial_punctuation_does_not_retry_or_drop_batch(monkeypatch):
    calls=[]; real=httpx.Client
    def respond(request):
        calls.append(request)
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'boundaries':[{'after':1,'mark':'の'},{'after':2,'mark':'。'}]})}}]})
    monkeypatch.setattr(httpx,'Client',lambda **kw:real(transport=httpx.MockTransport(respond),**kw))
    p=DeepSeekPunctuator(Settings())
    assert p.restore([Word('私',0,1,'A'),Word('です',1,4,'A')]) == '私です。'
    assert len(calls)==1 and len(p.warnings)==1 and '1 个无效建议' in p.warnings[0]


def test_long_text_reaches_synthesis_and_keeps_segment_mapping(tmp_path):
    sys.path.insert(0,'/app/tests')
    from test_pipeline import Separator, Transcriber, Translator, Synthesizer, Pipeline, SimpleNamespace, media
    storage=LocalStorage(str(tmp_path))
    media.ffmpeg(['-f','lavfi','-i','color=s=160x90:r=24:d=8','-f','lavfi','-i','sine=duration=8',
                  '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',storage.path('input.mp4')])
    class LongText(Synthesizer):
        def tokenize(self,texts):return [148]*len(texts)
    synth=LongText(storage)
    pipe=Pipeline(Settings(storage_root=str(tmp_path)),storage,Separator(storage),Transcriber(),Translator(),synth)
    output=pipe.run(SimpleNamespace(id='long',input_key='input.mp4',target_language='zh',terminology=''),lambda *_:None,lambda:None)
    manifest=storage.read_json(output['manifest'])
    assert len(synth.calls)==2
    assert all(s['render_status']=='translated' and not s['flags'] for s in manifest['segments'])
    assert all('synthesis_internal_chunks:148_tokens' in s['notes'] for s in manifest['segments'])
    assert all(s['speaker_reference']==s['emotion_reference']==s['original_audio'] for s in manifest['segments'])
    assert not any(s['speaker_id'] for s in manifest['segments'])


def test_no_recognized_text_preserves_original_dialogue(tmp_path):
    sys.path.insert(0,'/app/tests')
    from test_pipeline import Separator, Transcriber, Translator, Synthesizer, Pipeline, SimpleNamespace, media
    storage=LocalStorage(str(tmp_path))
    media.ffmpeg(['-f','lavfi','-i','color=s=160x90:r=24:d=8','-f','lavfi','-i','sine=duration=8',
                  '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',storage.path('input.mp4')])
    class NoText(Transcriber):
        def transcribe(self,audio):return {'text':'','language_code':'en','words':[]}
    synth=Synthesizer(storage)
    output=Pipeline(Settings(storage_root=str(tmp_path)),storage,Separator(storage),NoText(),Translator(),synth).run(
        SimpleNamespace(id='no-text',input_key='input.mp4',target_language='zh',terminology=''),lambda *_:None,lambda:None)
    manifest=storage.read_json(output['manifest'])
    original,_=sf.read(storage.path(manifest['stems']['dialogue']))
    from pathlib import PurePosixPath
    kept,_=sf.read(storage.path(str(PurePosixPath(output['audio']).parent/'dialogue-translated.wav')))
    assert np.array_equal(original,kept) and not synth.calls
