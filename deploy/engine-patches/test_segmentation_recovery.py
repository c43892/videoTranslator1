import json
import sys

import httpx
import numpy as np
import pytest
import soundfile as sf
from videotranslator import media
from videotranslator.config import Settings
from videotranslator.domain import Word, Segment, ProviderError, NeedsReview
from videotranslator.punctuation import DeepSeekPunctuator, indexed_punctuation, apply_punctuation, lexical
from videotranslator.segmentation import segment_dialogue, join_words
from videotranslator.storage import LocalStorage


def test_indexed_punctuation_cannot_modify_words_or_timing():
    words = [Word('今日',0,1,'A'), Word('こんにちは',1,4,'A'),Word('次',4,5,'A')]
    text = indexed_punctuation(words,[{'after':2,'mark':'。'}])
    restored = apply_punctuation(words,text)
    assert text == '今日こんにちは。次'
    assert lexical(text) == lexical(join_words(words))
    assert [(w.start,w.end,w.speaker_id) for w in restored] == [(w.start,w.end,w.speaker_id) for w in words]


@pytest.mark.parametrize('boundaries', [None, [{'after':0,'mark':'.'}], [{'after':2,'mark':'.'}],
    [{'after':True,'mark':'.'}], [{'after':1,'mark':'new words'}],
    [{'after':1,'mark':'.'},{'after':1,'mark':','}]])
def test_invalid_positions_or_text_are_rejected(boundaries):
    with pytest.raises(ProviderError):
        indexed_punctuation([Word('Hello',0,4,'A')],boundaries)


def test_bad_batch_retries_locally_and_does_not_discard_valid_neighbors(monkeypatch):
    requests=[]; original=httpx.Client
    def handle(request):
        tokens=json.loads(request.content)['messages'][1]['content']
        tokens=json.loads(tokens)['tokens']; requests.append(len(tokens))
        payload={'text':'Rewritten source'} if len(requests)==1 else {'boundaries':[{'after':len(tokens),'mark':'.'}]}
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(payload)}}]})
    monkeypatch.setattr(httpx,'Client',lambda **kw:original(transport=httpx.MockTransport(handle),**kw))
    words=[Word(text,i,i+1,'A') for i,text in enumerate(['One','two','three','four'])]
    p=DeepSeekPunctuator(Settings()); text=p.restore(words)
    assert requests==[4,2,2] and not p.warnings
    assert lexical(text)==lexical(join_words(words)) and 'two.' in text and 'four.' in text


def test_zero_time_word_joins_same_speaker_without_fabricating_time():
    words=[Word('コメント、',1,1,'A'),Word('お願いします。',1.2,5,'A')]
    segments=segment_dialogue(words)
    assert len(segments)==1 and segments[0].source_text=='コメント、お願いします。'
    assert (segments[0].start,segments[0].end)==(1,5) and not segments[0].flags


def test_zero_time_word_does_not_join_another_speaker():
    segments=segment_dialogue([Word('コメント、',1,1,'A'),Word('はい。',1.2,5,'B')])
    assert len(segments)==2 and 'zero_duration_utterance' in segments[0].flags
    assert segments[1].speaker_id=='B'


def test_zero_length_fallback_does_not_abort_other_translated_speech(tmp_path):
    storage=LocalStorage(str(tmp_path))
    sf.write(tmp_path/'original.wav',np.ones((4*48000,2))*.1,48000,subtype='FLOAT')
    sf.write(tmp_path/'dub.wav',np.ones((48000,2))*.3,48000,subtype='FLOAT')
    segments=[Segment('zero','A',1,1,'word',render_status='original'),
              Segment('good','B',2,3,'Translated',translation='译文',aligned_audio='dub.wav',render_status='translated')]
    media.dialogue_timeline(segments,storage,tmp_path/'result.wav',4,original=tmp_path/'original.wav')
    result,_=sf.read(tmp_path/'result.wav')
    assert np.allclose(result[:2*48000],0)
    assert np.allclose(result[2*48000:3*48000],.3)
    assert '00:00:01,000 --> 00:00:01,000' not in media.subtitles(segments,'source_text')


def test_all_original_fallback_is_not_reported_as_success(tmp_path):
    sys.path.insert(0,'/app/tests')
    from test_pipeline import Separator, Transcriber, Translator, Synthesizer, Pipeline, SimpleNamespace
    storage=LocalStorage(str(tmp_path))
    media.ffmpeg(['-f','lavfi','-i','color=s=160x90:r=24:d=8','-f','lavfi','-i','sine=duration=8',
                  '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',storage.path('input.mp4')])
    class Unknown(Transcriber):
        def transcribe(self,audio):
            t=super().transcribe(audio)
            for w in t['words']: w['speaker_id']='unknown'
            return t
    class UnusableSynthesis(Synthesizer):
        def synthesize(self,segment,output,language):
            raise NeedsReview('No usable generated audio')
    pipeline=Pipeline(Settings(storage_root=str(tmp_path)),storage,Separator(storage),Unknown(),Translator(),UnusableSynthesis(storage))
    with pytest.raises(NeedsReview,match='未生成可用的译制对白'):
        pipeline.run(SimpleNamespace(id='none',input_key='input.mp4',target_language='zh',terminology=''),lambda *_:None,lambda:None)
