"""Whisper word timing plus an independent speaker-labeling adapter.

Compression and reference-linked speaker request chunks are API transport
concerns, separate from semantic utterance segmentation.
"""
import math
import re
from pathlib import PurePosixPath
import httpx
from . import media
from .domain import ProviderError
from .providers import require_success
from .diarization import chunked_diarize

MAX_UPLOAD_BYTES = 24_000_000  # Leave room below OpenAI's 25 MB file limit.


def punctuated_words(transcript):
    """Restore punctuation omitted by Whisper word timestamps, without changing words."""
    words = transcript['words']
    text, cursor, matches = transcript.get('text', ''), 0, []
    for word in words:
        value = word.get('word')
        if not isinstance(value, str) or not value.strip():
            raise ProviderError('Whisper returned an invalid word')
        match = re.search(re.escape(value.strip()), text[cursor:], flags=re.IGNORECASE)
        if not match or any(c.isalnum() for c in text[cursor:cursor+match.start()]):
            return [w['word'] for w in words]
        matches.append((cursor+match.start(), cursor+match.end()))
        cursor += match.end()
    if any(c.isalnum() for c in text[cursor:]):
        return [w['word'] for w in words]
    return [text[left:(matches[i+1][0] if i+1<len(matches) else len(text))].strip()
            for i,(left,right) in enumerate(matches)]


def timed_interval(item, duration):
    try:
        start, end = float(item['start']), float(item['end'])
    except (KeyError, ValueError, TypeError) as exc:
        raise ProviderError('Transcription returned invalid timestamps') from exc
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start or end > duration + .25:
        raise ProviderError('Transcription timestamps are outside the source audio')
    if start > duration:
        raise ProviderError('Transcription timestamps are outside the source audio')
    return start, min(end, duration)


def attach_speakers(transcript, diarization, duration):
    if not isinstance(transcript.get('words'), list) or not isinstance(diarization.get('segments'), list):
        raise ProviderError('OpenAI response is missing word timing or speaker segments')
    # Whisper sometimes emits empty alignment entries between real words.
    # They carry no speech text and must not fail an otherwise valid transcript.
    # Keep invalid types as errors and preserve every nonempty word/timestamp.
    empty_words = sum(isinstance(w, dict) and isinstance(w.get('word'), str)
                      and not w['word'].strip() for w in transcript['words'])
    transcript = {**transcript, 'words': [w for w in transcript['words']
        if not (isinstance(w, dict) and isinstance(w.get('word'), str) and not w['word'].strip())]}
    turns = []
    for turn in diarization['segments']:
        start, end = timed_interval(turn, duration)
        speaker = turn.get('speaker')
        if not isinstance(speaker, str) or not speaker.strip():
            raise ProviderError('Speaker-labeling response contains a missing speaker ID')
        turns.append((start, end, speaker))
    words = []
    unresolved = 0
    boundary_tolerance = 0
    zero_duration = 0
    for word, text in zip(transcript['words'], punctuated_words(transcript)):
        start, end = timed_interval(word, duration)
        if not isinstance(text, str) or not text.strip():
            raise ProviderError('Whisper returned an invalid word')
        if end <= start:
            # Whisper can place a word at a point. Keep its text and original
            # timing: the surrounding utterance supplies the audio interval.
            candidates = {who for left, right, who in turns if left <= start <= right}
            if not candidates:
                candidates = {who for left, right, who in turns if left-.5 <= start <= right+.5}
            speaker = next(iter(candidates)) if len(candidates) == 1 else 'unknown'
            unresolved += speaker == 'unknown'
            zero_duration += 1
            words.append({'type': 'word', 'text': text, 'start': start, 'end': end, 'speaker_id': speaker})
            continue
        coverage = {}
        for left, right, speaker in turns:
            overlap = max(0, min(end, right)-max(start, left))
            if overlap:
                coverage[speaker] = coverage.get(speaker, 0) + overlap
        ranked = sorted(coverage, key=coverage.get, reverse=True)
        speaker = 'unknown'
        if ranked and coverage[ranked[0]] >= (end-start)*.5:
            # Conflicting speaker turns must not quietly become one cloned voice.
            if len(ranked) == 1 or coverage[ranked[1]] < (end-start)*.2:
                speaker = ranked[0]
        if speaker == 'unknown' and len(ranked) == 1:
            nearby = {who for left,right,who in turns if min(end+.5,right)>max(start-.5,left)}
            covered = coverage[ranked[0]]
            if nearby == {ranked[0]} and covered >= (end-start)*.25 and end-start-covered <= .5:
                # The two APIs estimate boundaries independently. Allow a small
                # edge discrepancy only when no competing speaker is nearby.
                speaker = ranked[0]
                boundary_tolerance += 1
        if speaker == 'unknown':
            unresolved += 1
        words.append({'type': 'word', 'text': text, 'start': start, 'end': end, 'speaker_id': speaker})
    if transcript.get('text', '').strip() and not words:
        raise ProviderError('Whisper returned speech without usable word timestamps')
    return {'text': transcript.get('text', ''), 'language_code': transcript.get('language', ''),
            'words': words, 'speaker_assignment': {'method': 'temporal_overlap', 'unresolved_words': unresolved,
                                                   'boundary_tolerance_words': boundary_tolerance,
                                                   'zero_duration_words': zero_duration, 'empty_alignment_entries': empty_words}}


class OpenAISpeakerDiarizer:
    """Replaceable speaker-labeling step; its text never replaces Whisper text."""
    def __init__(self, config, check_cancel=lambda: None):
        self.config = config
        self.check_cancel = check_cancel

    def diarize(self, path):
        return chunked_diarize(path, self.request, self.check_cancel)

    def request(self, path, references):
        data = {'model': self.config.diarization_model, 'response_format': 'diarized_json',
                'chunking_strategy': 'auto'}
        if references:
            data['known_speaker_names[]'] = list(references)
            data['known_speaker_references[]'] = list(references.values())
        with httpx.Client(timeout=httpx.Timeout(60, read=1800)) as client, path.open('rb') as audio:
            return require_success(client.post(self.config.openai_base_url.rstrip('/') + '/audio/transcriptions',
                headers={'Authorization': 'Bearer ' + self.config.openai_api_key},
                data=data, files={'file': ('dialogue.mp3', audio, 'audio/mpeg')}),
                'OpenAI speaker labeling')


class WhisperDiarizedTranscriber:
    use_diarization = True  # Legacy adapter retained for old callers only.
    def __init__(self, config, storage, diarizer=None, check_cancel=lambda: None):
        self.config, self.storage = config, storage
        self.diarizer = diarizer or OpenAISpeakerDiarizer(config, check_cancel)

    def prepare_audio(self, audio, key):
        path = self.storage.path(key)
        if self.storage.exists(key) and path.stat().st_size <= MAX_UPLOAD_BYTES:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.partial.mp3')
        # Speech transport only: keep full-quality source audio for references/mixing.
        # Start at 48 kbps; lower rates keep longer uploads under the byte limit.
        # This does not bypass a provider's separate audio-duration limit.
        for bitrate in (48, 32, 24):
            media.ffmpeg(media.input_args(self.storage.path(audio)) +
                ['-vn', '-ar', '16000', '-ac', '1', '-c:a', 'libmp3lame', '-b:a', f'{bitrate}k', temporary])
            if temporary.stat().st_size <= MAX_UPLOAD_BYTES:
                temporary.replace(path)
                return path
        temporary.unlink(missing_ok=True)
        raise ProviderError('Audio exceeds the OpenAI upload limit after compression')

    def transcribe(self, audio):
        prefix = str(PurePosixPath(audio).parent / 'recognition')
        prepared = self.prepare_audio(audio, prefix + '/dialogue.mp3')
        duration = float(media.probe(self.storage.path(audio))['format']['duration'])
        whisper_key, speakers_key = prefix+'/whisper.json', prefix+'/speakers.json'
        if self.storage.exists(whisper_key):
            transcript = self.storage.read_json(whisper_key)
        else:
            with httpx.Client(timeout=httpx.Timeout(60, read=1800)) as client, prepared.open('rb') as f:
                transcript = require_success(client.post(self.config.openai_base_url.rstrip('/') + '/audio/transcriptions',
                    headers={'Authorization': 'Bearer ' + self.config.openai_api_key},
                    data={'model': self.config.whisper_model, 'response_format': 'verbose_json',
                          'timestamp_granularities[]': ['word', 'segment'], 'temperature': '0'},
                    files={'file': ('dialogue.mp3', f, 'audio/mpeg')}), 'OpenAI Whisper')
            if not isinstance(transcript.get('words'), list):
                raise ProviderError('Whisper response has no word-level timestamps')
            self.storage.write_json(whisper_key, transcript)
        if not self.use_diarization:
            result = whisper_timed_words(transcript, duration)
            result['raw_responses'] = {'whisper': whisper_key}
            return result
        if self.storage.exists(speakers_key):
            diarization = self.storage.read_json(speakers_key)
        else:
            diarization = self.diarizer.diarize(prepared)
            if not isinstance(diarization.get('segments'), list):
                raise ProviderError('Speaker-labeling response has no speaker segments')
            self.storage.write_json(speakers_key, diarization)
        result = attach_speakers(transcript, diarization, duration)
        result['raw_responses'] = {'whisper': whisper_key, 'diarization': speakers_key}
        return result


def whisper_timed_words(transcript, duration):
    if not isinstance(transcript.get('words'), list):
        raise ProviderError('Whisper response has no word-level timestamps')
    transcript = {**transcript, 'words': [w for w in transcript['words']
        if not (isinstance(w, dict) and isinstance(w.get('word'), str) and not w['word'].strip())]}
    words = []
    for raw, text in zip(transcript['words'], punctuated_words(transcript)):
        start, end = timed_interval(raw, duration)
        words.append({'type':'word', 'text':text, 'start':start, 'end':end, 'speaker_id':''})
    if transcript.get('text', '').strip() and not words:
        raise ProviderError('Whisper returned speech without usable word timestamps')
    utterances = []
    for raw in transcript.get('segments', []):
        if not isinstance(raw, dict) or not isinstance(raw.get('text'), str) or not raw['text'].strip():
            continue
        start, end = timed_interval(raw, duration)
        utterances.append({'text': raw['text'].strip(), 'start': start, 'end': end})
    return {'text':transcript.get('text',''), 'language_code':transcript.get('language',''),
            'words':words, 'utterances':utterances, 'provider':'openai-whisper'}


class WhisperTranscriber(WhisperDiarizedTranscriber):
    use_diarization = False

    def __init__(self, config, storage, check_cancel=lambda:None):
        self.config, self.storage = config, storage
