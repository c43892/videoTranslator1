import json
import httpx
import pytest
from videotranslator.domain import Word, ProviderError
from videotranslator.punctuation import apply_punctuation, DeepSeekPunctuator
from videotranslator.segmentation import join_words, segment_dialogue
from videotranslator.config import Settings


def test_punctuation_preserves_source_and_original_word_timestamps():
    words=[Word('大',0,.4,'unknown'),Word('丈夫ですか',.4,1,'A'),Word('行きましょう',1.1,3.5,'A')]
    restored=apply_punctuation(words,'大丈夫ですか？行きましょう。')
    assert [(w.start,w.end,w.speaker_id) for w in restored]==[(w.start,w.end,w.speaker_id) for w in words]
    assert words[-1].text=='行きましょう'
    assert join_words(restored)=='大丈夫ですか？行きましょう。'
    assert len(segment_dialogue(restored))==2


@pytest.mark.parametrize('text',['你没事吧？','大夫ですか？行きましょう。','行きましょう。大丈夫ですか？'])
def test_rewritten_translated_or_reordered_words_are_rejected(text):
    with pytest.raises(ProviderError):
        apply_punctuation([Word('大丈夫ですか',0,1,'A'),Word('行きましょう',1,4,'A')],text)


def test_punctuation_adapter_requests_original_language_only(monkeypatch):
    original=httpx.Client
    def handle(request):
        payload=json.loads(request.content)
        assert 'Do not translate' in payload['messages'][0]['content']
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop',
            'message':{'content':json.dumps({'text':'Hello, world.'})}}]})
    monkeypatch.setattr(httpx,'Client',lambda **kw:original(transport=httpx.MockTransport(handle),**kw))
    assert DeepSeekPunctuator(Settings()).restore([Word('Hello',0,1,'A'),Word('world',1,4,'A')])=='Hello, world.'


@pytest.mark.parametrize('content,finish', [('{"text":"Changed words."}', 'stop'), ('broken JSON', 'stop'), ('{}', 'length')])
def test_bad_optional_punctuation_preserves_source_instead_of_failing(monkeypatch,content,finish):
    original=httpx.Client
    monkeypatch.setattr(httpx,'Client',lambda **kw:original(transport=httpx.MockTransport(
        lambda request:httpx.Response(200,json={'choices':[{'finish_reason':finish,'message':{'content':content}}]})),**kw))
    words=[Word('Hello,',0,1,'A'),Word('world.',1,4,'A')]
    punctuator=DeepSeekPunctuator(Settings())
    assert punctuator.restore(words)=='Hello, world.'
    assert len(punctuator.warnings)==1
    assert [(w.start,w.end) for w in apply_punctuation(words,punctuator.restore(words))]==[(0,1),(1,4)]
    assert len(punctuator.warnings)==1


def test_provider_auth_failure_is_not_hidden_as_punctuation_fallback(monkeypatch):
    original=httpx.Client
    monkeypatch.setattr(httpx,'Client',lambda **kw:original(transport=httpx.MockTransport(
        lambda request:httpx.Response(401)),**kw))
    with pytest.raises(ProviderError):
        DeepSeekPunctuator(Settings()).restore([Word('Hello.',0,4,'A')])
