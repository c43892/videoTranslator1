import json

import httpx
import pytest

from videotranslator.adapters.conversation import DeepSeekConversationInterpreter, GuidedInterpreter
from .test_conversations import AUTH, URL, client, create, edit


@pytest.mark.parametrize('message', ['配音效果太差了', '这个功能不好用', '翻译不准确，我很失望', 'The dubbing sounds unnatural'])
def test_product_complaint_acknowledges_without_email_or_changing_slots(message):
    result=GuidedInterpreter().interpret(message, {'locale':'zh','target_language':'en','source_kind':'youtube'})
    assert result.intent=='complaint' and 'c43892@gmail.com' not in result.reply
    assert not result.target_language and not result.source_kind and not result.youtube_url


@pytest.mark.parametrize('message', ['翻译成法语', '怎么上传视频', '陪我聊聊天', '今天心情很差'])
def test_ordinary_or_unrelated_messages_do_not_advertise_contact(message):
    result=GuidedInterpreter().interpret(message, {'locale':'zh'})
    assert 'c43892@gmail.com' not in result.reply


def test_complaint_preserves_existing_draft(client,container):
    draft=edit(client,create(client,'zh'),text=f'翻译成英文 {URL}')
    feedback=edit(client,draft,text='这个功能不好用')
    assert feedback['ready'] and feedback['target_language']=='en' and feedback['youtube_url']==URL
    assert 'c43892@gmail.com' not in feedback['messages'][-1]['text']
    assert feedback['status']=='draft' and not feedback.get('job_id')


@pytest.mark.parametrize('message', ['翻译成法语', '法语', '改成日语', 'Translate this video into French', 'Actually make it French'])
def test_unsupported_target_explained_even_without_ai(message):
    result = GuidedInterpreter().interpret(message, {'locale':'zh','explicit_locale':'zh'})
    assert result.target_language == 'unsupported'
    assert '中文或英文' in result.reply
    assert result.intent == 'product'


@pytest.mark.parametrize('message', ['把法语视频翻译成英文', 'Translate this French video into English'])
def test_source_language_is_not_an_unsupported_target(message):
    result = GuidedInterpreter().interpret(message, {})
    assert result.target_language == 'en'


@pytest.mark.parametrize('message', ['陪我聊聊天', '讲一个笑话', '帮我写一首诗', 'What is the weather today?', 'Hello'])
def test_guided_mode_politely_declines_unrelated_requests(message):
    result = GuidedInterpreter().interpret(message, {'locale':'zh'})
    assert result.intent == 'off_topic' and result.reply
    assert not result.source_kind and not result.target_language


def test_capability_question_does_not_clear_current_target():
    result = GuidedInterpreter().interpret('支持法语吗？', {'locale':'zh','target_language':'en'})
    assert result.intent == 'product' and not result.target_language
    assert '中文或英文' in result.reply


def test_ai_outage_still_explains_french_and_preserves_explicit_ui_language(monkeypatch):
    def offline(*args, **kwargs): raise httpx.ConnectError('offline')
    monkeypatch.setattr(httpx,'post',offline)
    result = DeepSeekConversationInterpreter('test','test','https://example.invalid').interpret(
        '翻译成法语', {'locale':'en','explicit_locale':'en'})
    assert result.mode == 'fallback' and result.target_language == 'unsupported'
    assert 'Chinese or English' in result.reply


def test_ai_off_topic_classification_cannot_change_task_slots(monkeypatch):
    def answer(*args, **kwargs):
        return httpx.Response(200,request=httpx.Request('POST','https://example.invalid'),json={
            'choices':[{'message':{'content':json.dumps({'intent':'off_topic','detected_locale':'zh',
                'source_kind':'upload','target_language':'en','reply':'A joke that should not be shown'})}}]})
    monkeypatch.setattr(httpx,'post',answer)
    result=DeepSeekConversationInterpreter('test','test','https://example.invalid').interpret('讲个笑话',{})
    assert not result.source_kind and not result.target_language
    assert '抱歉' in result.reply and '笑话' not in result.reply


def test_unsupported_and_off_topic_messages_do_not_charge_or_submit(client,container):
    draft=edit(client,create(client,'zh'),text=f'翻译成英文 {URL}')
    assert draft['ready']
    refused=edit(client,draft,text='陪我聊聊天')
    assert refused['target_language']=='en' and refused['youtube_url']==URL
    assert '暂不提供闲聊' in refused['messages'][-1]['text']
    changed=edit(client,refused,text='改成法语')
    assert not changed['ready'] and changed['target_language']==''
    assert '中文或英文' in changed['messages'][-1]['text']
    from videotranslator.domain.models import Job,LedgerEntry
    with container.store.transaction() as tx:
        assert not tx.query(Job) and not tx.query(LedgerEntry)
