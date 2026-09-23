"""Recognize a completed job's cached dialogue without changing its result/billing."""
import json
import sys
import time
from collections import Counter
from videotranslator.config import settings
from videotranslator.storage import LocalStorage
from videotranslator.scribe import create_transcriber
from videotranslator.segmentation import parse_words, segment_dialogue


def main(job_id):
    config = settings()
    assert config.transcription_provider == 'scribe'
    store = LocalStorage(config.storage_root)
    manifest = store.read_json(f'jobs/{job_id}/manifest.json')
    started = time.monotonic()
    transcript = create_transcriber(config, store).transcribe(manifest['stems']['dialogue'])
    events = transcript['audio_events']
    segments = segment_dialogue(parse_words(transcript), audio_events=events,
                               turn_boundaries=transcript.get('turn_boundaries', []))
    # An event wholly between words must not be swallowed by a merged utterance.
    words = parse_words(transcript)
    isolated = [e for e in events if not any(w.start < e['end'] and w.end > e['start'] for w in words)]
    swallowed = [(s.id, e) for s in segments for e in isolated
                 if s.start < e['start'] and s.end > e['end']]
    assert not swallowed, 'Segmentation swallowed a separate sound event'
    assert all(not s.speaker_id for s in segments)
    prefix = f'jobs/{job_id}/validation/scribe-v2'
    store.write_json(prefix+'/transcript.json', transcript)
    store.write_json(prefix+'/segments.json', {'segments':[s.to_dict() for s in segments]})
    summary = {'model':config.scribe_model, 'duration':manifest['duration'],
        'elapsed_seconds':round(time.monotonic()-started, 2),
        'language':transcript['language_code'], 'words':len(words), 'segments':len(segments),
        'audio_events':len(events), 'isolated_events':len(isolated),
        'turn_boundaries':transcript.get('turn_boundaries', []),
        'event_types':dict(Counter(e['text'] for e in events)),
        'event_examples':events[:8], 'speaker_identification':False,
        'existing_result_changed':False}
    store.write_json(prefix+'/summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
