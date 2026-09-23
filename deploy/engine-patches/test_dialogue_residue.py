import numpy as np
import pytest
import soundfile as sf
from videotranslator import media
from videotranslator.domain import Segment
from videotranslator.storage import LocalStorage


def render(tmp_path, segments, original=True):
    storage = LocalStorage(str(tmp_path))
    # Nonzero original audio everywhere exposes leakage at all uncovered times.
    sf.write(tmp_path/'original.wav', np.full((4*48000, 2), .1), 48000, subtype='FLOAT')
    sf.write(tmp_path/'dub.wav', np.full((48000, 2), .3), 48000, subtype='FLOAT')
    media.dialogue_timeline(segments, storage, tmp_path/'out.wav', 4,
                           original=tmp_path/'original.wav' if original else None)
    return sf.read(tmp_path/'out.wav')[0]


def translated(start=2):
    return Segment('good','speaker',start,start+1,'Text',aligned_audio='dub.wav',render_status='translated')


def test_no_original_voice_before_after_or_between_translations(tmp_path):
    audio = render(tmp_path, [translated()])
    assert np.all(audio[:2*48000] == 0)
    assert np.allclose(audio[2*48000:3*48000], .3)
    assert np.all(audio[3*48000:] == 0)


def test_only_explicit_fallback_is_preserved_without_doubling_overlaps(tmp_path):
    audio = render(tmp_path, [Segment('a','s',0,.7,'A',render_status='original'),
        Segment('b','s',.5,1,'B',render_status='original'), translated()])
    assert np.allclose(audio[:48000], .1)
    assert np.all(audio[48000:2*48000] == 0)
    assert np.allclose(audio[2*48000:3*48000], .3)
    assert np.all(audio[3*48000:] == 0)


def test_translated_overlapping_speakers_still_accumulate(tmp_path):
    audio = render(tmp_path, [translated(1), translated(1.5)])
    assert np.allclose(audio[72000:96000], .6)
    assert np.all(audio[:48000] == 0)


def test_fallback_without_original_is_an_error(tmp_path):
    with pytest.raises(ValueError, match='required'):
        render(tmp_path, [Segment('a','s',0,1,'A',render_status='original')], original=False)
