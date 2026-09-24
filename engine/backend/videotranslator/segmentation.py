"""Split at source-language punctuation, then merge short neighboring clauses."""
import re
import math
from dataclasses import replace
from .domain import Word, Segment

# Ignore decimal points and common English titles; include Japanese punctuation.
BOUNDARY = re.compile(r'[,;:!?，；：、。！？…][”’"\'」』）)]?$|(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bMrs)[.][”’"\'」』）)]?$')
CJK = re.compile(r'[\u3000-\u30ff\u3400-\u9fff\uff66-\uff9f]')
SEGMENTATION_VERSION = 'punctuation-v5-sentence-boundaries'
SENTENCE_END = re.compile(r'[.!?。！？…][”’"\'」』）)]?$')
MARKS = set('.,!?;:，。！？；：、…“”‘’「」『』（）()"')


def join_words(words: list[Word]) -> str:
    text = ''
    for w in words:
        piece = w.text.strip()
        if not piece:
            continue
        if text and not CJK.search(text[-1]) and not CJK.search(piece[0]) and piece[0] not in '.,!?;:，。！？；：、':
            text += ' '
        text += piece
    return text


def parse_words(transcript: dict) -> list[Word]:
    words = []
    for raw in transcript.get('words', []):
        if raw.get('type', 'word') != 'word' or raw.get('start') is None or raw.get('end') is None:
            continue
        start, end = float(raw['start']), float(raw['end'])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            continue
        words.append(Word(raw['text'], start, end, raw.get('speaker_id') or 'unknown'))
    return sorted(words, key=lambda w: (w.start, w.end))


def speaker_ids(group):
    return {w.speaker_id for w in group if w.speaker_id != 'unknown'}


def span(group):
    return max(w.end for w in group)-group[0].start


def segment_dialogue(words: list[Word], max_reference_seconds: float = 15, audio_events=None, turn_boundaries=None) -> list[Segment]:
    # Scribe can return punctuation as separate timed tokens. It belongs to
    # the preceding text, never to a new audio window or the following voice.
    cleaned = []
    for word in words:
        token = word.text.strip()
        if not token:
            continue
        if all(c in MARKS for c in token):
            if cleaned:
                cleaned[-1] = replace(cleaned[-1], text=cleaned[-1].text + token)
            continue
        cleaned.append(word)
    words = cleaned
    events = audio_events or []
    def event_between(left, right):
        return (any(e['start'] < right[0].start and e['end'] > left[-1].end for e in events)
                or any(left[0].start < t <= right[0].start for t in (turn_boundaries or [])))
    clauses, current = [], []
    for word in words:
        if current and event_between(current, [word]):
            clauses.append(current)
            current = []
        current.append(word)
        if BOUNDARY.search(word.text.strip()):
            clauses.append(current)
            current = []
    if current:
        clauses.append(current)

    # Prefer the next clause; merge a short tail backward. No identity inference:
    # every resulting segment conditions its own voice.
    groups = list(clauses)
    i = 0
    while i < len(groups):
        if span(groups[i]) > 3:
            i += 1
            continue
        candidates = [j for j in (i+1, i-1) if 0 <= j < len(groups)]
        merged = False
        for j in candidates:
            left, right = sorted((i,j))
            if event_between(groups[left], groups[right]):
                continue
            # A complete short answer must not absorb the next person's turn.
            # We have no evidence of voice continuity across a sentence or pause.
            if SENTENCE_END.search(groups[left][-1].text.strip()):
                continue
            if groups[right][0].start - max(w.end for w in groups[left]) > .3:
                continue
            combined = groups[left] + groups[right]
            groups[left:right+1] = [combined]
            i = left
            merged = True
            break
        if not merged:
            i += 1

    result = []
    for i, group in enumerate(groups):
        flags, notes = [], []
        if span(group) <= 0:
            flags.append('zero_duration_utterance')
        if span(group) > max_reference_seconds + .001:
            notes.append('long_utterance_requires_bounded_emotion_reference')
        if span(group) <= 3:
            notes.append('short_utterance_no_compatible_neighbor')
        # Empty compatibility field; no identity assignment or speaker bank.
        result.append(Segment(f'seg-{i+1:05d}', '', group[0].start,
                              max(w.end for w in group), join_words(group), flags=flags, notes=notes))
    return result
