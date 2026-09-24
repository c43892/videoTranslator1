"""Bounded API requests with reference-based identity across transport chunks."""
import base64
import hashlib
import json
import math
from . import media
from .domain import ProviderError
from .storage import LocalStorage

VERSION = 'references-overlap-v1'
CHUNK_SECONDS = 600
OVERLAP_SECONDS = 5
SINGLE_REQUEST_SECONDS = 1200  # Margin below the observed 1400-second API limit.


def validated_turns(result, duration, offset=0):
    if not isinstance(result.get('segments'), list):
        raise ProviderError('Speaker-labeling response has no speaker segments')
    turns = []
    for item in result['segments']:
        try:
            start, end = float(item['start']), float(item['end'])
            speaker = item['speaker']
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError('Invalid speaker segment') from exc
        if (not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start
                or end > duration + .25 or start > duration
                or not isinstance(speaker, str) or not speaker.strip()):
            raise ProviderError('Invalid speaker segment timing or identity')
        if end > start:
            turns.append(dict(item, start=start+offset, end=min(end, duration)+offset))
    return turns


def identity_map(turns, previous, references, known):
    """Never equate independently assigned A/B labels between requests."""
    mapping, warnings = {}, []
    prior_known = set(known)
    labels = list(dict.fromkeys(t['speaker'] for t in turns))
    for label in labels:
        scores = {}
        for turn in (t for t in turns if t['speaker'] == label):
            for prior in previous:
                overlap = min(turn['end'], prior['end'])-max(turn['start'], prior['start'])
                if overlap > 0 and prior['speaker'] != 'unknown':
                    who = prior['speaker']
                    scores[who] = scores.get(who, 0)+overlap
        ranked = sorted(scores, key=scores.get, reverse=True)
        match = None
        if ranked and scores[ranked[0]] >= 1.5 and scores[ranked[0]] >= .8*sum(scores.values()):
            match = ranked[0]
        if label in references:
            mapping[label] = label if match in (None, label) else 'unknown'
        elif match:
            mapping[label] = match
        elif scores or (prior_known and not prior_known.issubset(references)):
            # A returning speaker without a usable supplied reference cannot be
            # distinguished reliably from a new person. Preserve their speech.
            mapping[label] = 'unknown'
        else:
            mapping[label] = f'speaker_{len(known)+1:03d}'
            known.add(mapping[label])
        if mapping[label] == 'unknown':
            warnings.append(f'Unresolved cross-chunk speaker: {label}')
    # Two local identities must not silently collapse through overlap matching.
    for who in set(mapping.values())-{'unknown'}:
        aliases = [label for label, target in mapping.items() if target == who]
        if len(aliases) > 1:
            for label in aliases:
                mapping[label] = 'unknown'
            warnings.append(f'Conflicting cross-chunk speaker matches: {who}')
    return mapping, warnings


def chunked_diarize(path, request, check_cancel=lambda: None):
    duration = float(media.probe(path)['format']['duration'])
    if duration <= SINGLE_REQUEST_SECONDS:
        check_cancel()
        return request(path, {})
    cache = LocalStorage(str(path.parent / ('diarization-'+VERSION)))
    bank, known, previous, merged, chunks, warnings = {}, set(), [], [], [], []
    for index, start in enumerate(range(0, math.ceil(duration), CHUNK_SECONDS)):
        check_cancel()
        end = min(duration, start+CHUNK_SECONDS)
        left, right = max(0, start-OVERLAP_SECONDS), min(duration, end+OVERLAP_SECONDS)
        audio = cache.path(f'chunk-{index:03d}.mp3')
        if not audio.exists():
            partial = audio.with_suffix('.partial.mp3')
            media.ffmpeg(media.input_args(path) + ['-ss', f'{left:.6f}', '-t', f'{right-left:.6f}',
                '-vn', '-ar', '16000', '-ac', '1', '-c:a', 'libmp3lame', '-b:a', '48k', partial])
            partial.replace(audio)
        references = dict(list(bank.items())[:4])
        # Include reference identity/content in the checkpoint so retries cannot
        # apply a response to a different set of supplied speaker prompts.
        signature = hashlib.sha256(json.dumps(references, sort_keys=True).encode()).hexdigest()[:16]
        response_key = f'chunk-{index:03d}-{signature}.json'
        if cache.exists(response_key):
            result = cache.read_json(response_key)
        else:
            result = request(audio, references)
            validated_turns(result, right-left)
            cache.write_json(response_key, result)
        check_cancel()
        turns = validated_turns(result, right-left, left)
        # First chunk defines identities; later chunks use references/overlap.
        mapping, issues = identity_map(turns, previous, references, known)
        warnings.extend(f'chunk {index}: {issue}' for issue in issues)
        absolute = [dict(t, speaker=mapping[t['speaker']]) for t in turns]
        for turn in absolute:
            a, b = max(start, turn['start']), min(end, turn['end'])
            if b > a:
                merged.append(dict(turn, start=a, end=b))
        # Stable 2–8 second, non-overlapping speech clips; never use mixed voices
        # as identity references. Keep the first suitable clip for each speaker.
        for who in sorted(known):
            if who in bank or len(bank) >= 4:
                continue
            candidates = [t for t in absolute if t['speaker'] == who and t['end']-t['start'] >= 2.5
                and not any(o['speaker'] != who and min(t['end'],o['end']) > max(t['start'],o['start'])
                            for o in absolute)]
            for turn in sorted(candidates, key=lambda t: t['end']-t['start'], reverse=True):
                ref = cache.path(who+'.wav')
                a, b = turn['start']+.2, min(turn['end']-.2, turn['start']+8.2)
                media.ffmpeg(media.input_args(path) + ['-ss', f'{a:.6f}', '-t', f'{b-a:.6f}',
                    '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', ref])
                quality = media.quality(ref)
                if quality['rms'] >= .0001 and quality['clipped_fraction'] <= .05:
                    bank[who] = 'data:audio/wav;base64,'+base64.b64encode(ref.read_bytes()).decode()
                    break
        previous = absolute
        chunks.append({'start': start, 'end': end, 'request_start': left, 'request_end': right,
                       'response': str(cache.path(response_key)), 'speaker_map': mapping,
                       'reference_speakers': list(references)})
        cache.write_json('progress.json', {'completed_chunks': index+1,
            'total_chunks': math.ceil(duration/CHUNK_SECONDS), 'speakers': sorted(known), 'warnings': warnings})
    return {'segments': sorted(merged, key=lambda t: (t['start'], t['end'])),
            'strategy': VERSION, 'chunks': chunks, 'warnings': warnings,
            'reference_speakers': list(bank)}
