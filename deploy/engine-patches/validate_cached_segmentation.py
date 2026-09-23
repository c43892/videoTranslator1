"""Validate new segmentation from a failed task's cached ASR; no job or ledger changes."""
from collections import Counter
from pathlib import PurePosixPath
import json
import sys

from videotranslator.config import Settings
from videotranslator.storage import LocalStorage
from videotranslator.segmentation import parse_words, segment_dialogue, join_words, SEGMENTATION_VERSION
from videotranslator.punctuation import DeepSeekPunctuator, apply_punctuation, lexical


def validate(job_id):
    config=Settings(); storage=LocalStorage(config.storage_root)
    manifest=storage.read_json(f'jobs/{job_id}/manifest.json')
    words=parse_words(storage.read_json(manifest['transcript']))
    prefix=str(PurePosixPath(manifest['transcript']).parent/SEGMENTATION_VERSION)
    key=prefix+'/punctuation.json'
    if not storage.exists(key):
        punctuator=DeepSeekPunctuator(config)
        text=punctuator.restore(words)
        apply_punctuation(words,text)
        storage.write_json(key,{'text':text,'warnings':punctuator.warnings})
    restored=apply_punctuation(words,storage.read_json(key)['text'])
    segments=segment_dialogue(restored,config.emotion_reference_max_seconds)
    assert lexical(''.join(s.source_text for s in segments)) == lexical(join_words(words))
    assert [(w.start,w.end,w.speaker_id) for w in restored] == [(w.start,w.end,w.speaker_id) for w in words]
    summary={'job_id':job_id,'version':SEGMENTATION_VERSION,'words':len(words),'segments':len(segments),
        'source_text_preserved':True,'word_timestamps_preserved':True,
        'flags':dict(Counter(f for s in segments for f in s.flags)),
        'max_duration':max(s.end-s.start for s in segments),
        'speaker_reference_candidates':dict(Counter(s.speaker_id for s in segments if not s.flags
            and config.speaker_reference_min_seconds <= s.end-s.start <= config.speaker_reference_max_seconds)),
        'punctuation_warnings':storage.read_json(key)['warnings']}
    storage.write_json(prefix+'/segmentation-validation.json',{'summary':summary,'segments':[s.to_dict() for s in segments]})
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__': validate(sys.argv[1])
