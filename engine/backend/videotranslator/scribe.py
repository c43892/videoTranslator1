"""ElevenLabs Scribe adapter: words and sound events, without speaker identity."""
from pathlib import PurePosixPath
import hashlib
import json
import httpx
from . import media
from .domain import ProviderError
from .segmentation import join_words, parse_words
from .transcription import timed_interval, WhisperTranscriber


def normalize_scribe(data, duration):
    if not isinstance(data, dict) or not isinstance(data.get('words'), list):
        raise ProviderError('Scribe response has no timed words/events')
    words, events, turns = [], [], []
    previous_voice = None
    for item in data['words']:
        if not isinstance(item, dict):
            raise ProviderError('Scribe returned an invalid transcript entry')
        kind = item.get('type')
        if kind == 'spacing':
            continue
        if kind not in ('word', 'audio_event'):
            raise ProviderError('Scribe returned an unknown entry type')
        text = item.get('text')
        if not isinstance(text, str):
            raise ProviderError('Scribe returned invalid text')
        if not text.strip():
            continue
        start, end = timed_interval(item, duration)
        entry = {'text':text, 'start':start, 'end':end, 'type':kind}
        if kind == 'audio_event':
            events.append(entry)
        else:
            # Anonymous provider labels only detect a change in adjacent voices.
            # Do not expose identities, choose shared references or veto speech.
            voice = item.get('speaker_id')
            lexical = any(c.isalnum() for c in text)
            if lexical and voice is not None:
                if previous_voice is not None and voice != previous_voice:
                    turns.append(start)
                previous_voice = voice
            words.append(dict(entry, speaker_id=''))
    result = {'words':words, 'audio_events':events, 'turn_boundaries':turns, 'language_code':data.get('language_code',''),
              'provider':'elevenlabs', 'speaker_identification':False, 'turn_detection':'boundary_only'}
    # The provider's full transcript can include '(laughter)'. Only actual words
    # belong in translation or punctuation requests. Raw JSON remains cached.
    result['text'] = join_words(parse_words(result))
    return result


class ScribeTranscriber:
    def __init__(self, config, storage, check_cancel=lambda:None):
        self.config, self.storage, self.check_cancel = config, storage, check_cancel

    def transcribe(self, audio):
        self.check_cancel()
        if not self.config.elevenlabs_api_key.strip():
            raise ProviderError('Missing configuration: ELEVENLABS_API_KEY')
        signature=hashlib.sha256(json.dumps({'model':self.config.scribe_model,
            'base_url':self.config.elevenlabs_base_url, 'events':True, 'diarize':True}).encode()).hexdigest()[:16]
        prefix=str(PurePosixPath(audio).parent/'recognition')
        key=f'{prefix}/scribe-{signature}.json'
        duration=float(media.probe(self.storage.path(audio))['format']['duration'])
        if self.storage.exists(key):
            data=self.storage.read_json(key)
        else:
            prepared=self.storage.path(f'{prefix}/scribe-input.flac')
            if not prepared.exists():
                prepared.parent.mkdir(parents=True,exist_ok=True)
                partial=prepared.with_suffix('.partial.flac')
                media.ffmpeg(media.input_args(self.storage.path(audio))+['-vn','-ar','16000','-ac','1','-c:a','flac',partial])
                partial.replace(prepared)
            self.check_cancel()
            with httpx.Client(timeout=httpx.Timeout(60,read=1800)) as client, prepared.open('rb') as f:
                response=client.post(self.config.elevenlabs_base_url.rstrip('/')+'/speech-to-text',
                    headers={'xi-api-key':self.config.elevenlabs_api_key},
                    data={'model_id':self.config.scribe_model, 'timestamps_granularity':'word',
                          'tag_audio_events':'true', 'diarize':'true', 'no_verbatim':'false'},
                    files={'file':('dialogue.flac',f,'audio/flac')})
            # Do not log provider response bodies or authentication headers.
            if response.status_code in (401,403):
                raise ProviderError('ElevenLabs authentication/permission failed; check ELEVENLABS_API_KEY')
            if response.status_code==402:
                raise ProviderError('ElevenLabs account has insufficient transcription credit')
            if not response.is_success:
                raise ProviderError(f'ElevenLabs Scribe request failed (HTTP {response.status_code})')
            try:
                data=response.json()
            except ValueError as exc:
                raise ProviderError('ElevenLabs returned invalid JSON') from exc
            normalize_scribe(data,duration)
            self.storage.write_json(key,data)
        self.check_cancel()
        result=normalize_scribe(data,duration)
        result['raw_responses']={'scribe':key}
        return result


def create_transcriber(config, storage, check_cancel=lambda:None):
    cls=ScribeTranscriber if config.transcription_provider=='scribe' else WhisperTranscriber
    return cls(config, storage, check_cancel=check_cancel)
