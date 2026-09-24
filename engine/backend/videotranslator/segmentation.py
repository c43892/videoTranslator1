"""Build editing units using the legacy Whisper subtitle policy."""
import re
import math
from dataclasses import replace
from .domain import Word, Segment

# Ignore decimal points and common English titles; include Japanese punctuation.
BOUNDARY = re.compile(r'[,;:!?，；：、。！？…][”’"\'」』）)]?$|(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bMrs)[.][”’"\'」』）)]?$')
CJK = re.compile(r'[\u3000-\u30ff\u3400-\u9fff\uff66-\uff9f]')
SEGMENTATION_VERSION = 'legacy-whisper-utterances-v1'
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


def _event_between(left, right, events, turns):
    return (any(e['start'] < right[0].start and e['end'] > left[-1].end for e in events)
            or any(left[0].start < t <= right[0].start for t in turns))


def _provider_groups(words, utterances):
    """Assign every timed word once to the provider's ordered utterance spans."""
    if not utterances:
        return []
    valid = sorted((u for u in utterances if isinstance(u, dict) and u.get('start') is not None
                    and u.get('end') is not None and float(u['end']) >= float(u['start'])),
                   key=lambda u: (float(u['start']), float(u['end'])))
    if not valid:
        return []
    groups, cursor = [], 0
    for index, utterance in enumerate(valid):
        boundary = float(valid[index+1]['start']) if index+1 < len(valid) else math.inf
        group = []
        while cursor < len(words) and words[cursor].start < boundary:
            group.append(words[cursor]); cursor += 1
        if group:
            # The legacy editor used Whisper's segment interval as the audio
            # window. Word timestamps remain useful for text ordering, but are
            # not promoted to edit boundaries.
            start, end = float(utterance['start']), float(utterance['end'])
            group[0] = replace(group[0], start=start)
            group[-1] = replace(group[-1], end=max(end, group[-1].start))
            groups.append(group)
    if cursor < len(words):
        if groups:
            groups[-1].extend(words[cursor:])
        else:
            groups.append(words[cursor:])
    return groups


def _punctuation_groups(words, events, turns):
    groups, current = [], []
    for word in words:
        if current and _event_between(current, [word], events, turns):
            groups.append(current); current = []
        current.append(word)
        # Legacy Whisper editing units ended at complete sentences. Commas and
        # colons stayed inside the utterance instead of creating tiny TTS clips.
        if SENTENCE_END.search(word.text.strip()):
            groups.append(current); current = []
    if current:
        groups.append(current)
    return groups


def _current_groups(words, events, turns):
    """Retain the existing non-Whisper/Scribe word-level behavior."""
    clauses, current = [], []
    for word in words:
        if current and _event_between(current, [word], events, turns):
            clauses.append(current); current = []
        current.append(word)
        if BOUNDARY.search(word.text.strip()):
            clauses.append(current); current = []
    if current:
        clauses.append(current)
    groups = list(clauses)
    i = 0
    while i < len(groups):
        if span(groups[i]) > 3:
            i += 1; continue
        candidates = [j for j in (i+1, i-1) if 0 <= j < len(groups)]
        merged = False
        for j in candidates:
            left, right = sorted((i,j))
            if _event_between(groups[left], groups[right], events, turns):
                continue
            if SENTENCE_END.search(groups[left][-1].text.strip()):
                continue
            if groups[right][0].start-max(w.end for w in groups[left]) > .3:
                continue
            groups[left:right+1] = [groups[left]+groups[right]]
            i, merged = left, True
            break
        if not merged:
            i += 1
    return groups


def _legacy_sentence_groups(groups, events, turns, max_duration=7.0):
    """Mirror the old complete-sentence pass over Whisper utterances."""
    merged, current = [], []
    for group in groups:
        if not current:
            current = list(group); continue
        duration = max(w.end for w in current)-current[0].start
        combined_duration = max(w.end for w in group)-current[0].start
        text = join_words(current).rstrip()
        strong = bool(SENTENCE_END.search(text))
        medium = text.endswith((':', ';', '；', '：')) and len(text) > 30
        comma = text.endswith((',', '、', '，')) and len(text) > 50
        split = (strong or medium or comma or combined_duration >= max_duration
                 or (duration >= 3 and (medium or comma))
                 or _event_between(current, group, events, turns))
        if split:
            merged.append(current); current = list(group)
        else:
            current.extend(group)
    if current:
        merged.append(current)
    return merged


def _merge_close_groups(groups, events, turns, max_gap=.2, max_duration=10.0):
    """Mirror legacy merge_close_subtitles(min_gap_ms=200, max=10s)."""
    if not groups:
        return []
    merged, current = [], list(groups[0])
    for group in groups[1:]:
        gap = group[0].start-max(w.end for w in current)
        duration = max(w.end for w in group)-current[0].start
        if (gap < max_gap and duration <= max_duration
                and not _event_between(current, group, events, turns)):
            current.extend(group)
        else:
            merged.append(current); current = list(group)
    merged.append(current)
    return merged


def segment_dialogue(words: list[Word], max_reference_seconds: float = 15, audio_events=None,
                     turn_boundaries=None, utterances=None) -> list[Segment]:
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
    events, turns = audio_events or [], turn_boundaries or []
    groups = _provider_groups(words, utterances)
    if groups:
        groups = _legacy_sentence_groups(groups, events, turns)
        groups = _merge_close_groups(groups, events, turns)
    else:
        groups = _current_groups(words, events, turns)

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
