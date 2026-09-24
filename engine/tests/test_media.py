import numpy as np
import soundfile as sf
import pytest
from videotranslator import media
from videotranslator.domain import Segment, NeedsReview
from videotranslator.storage import LocalStorage

def test_short_speech_pads_without_slowing(tmp_path):
    source, output = tmp_path/'source.wav', tmp_path/'aligned.wav'
    signal = np.sin(np.arange(24000)*2*np.pi*440/48000)*.2
    sf.write(source, signal, 48000)
    media.align(source, output, 1, 1.25)
    data, rate = sf.read(output)
    assert rate == 48000 and len(data) == 48000
    assert np.max(np.abs(data[25000:])) < .001

def test_excessive_duration_is_compressed(tmp_path):
    source = tmp_path/'source.wav'
    sf.write(source, np.ones(96000)*.1, 48000)
    media.align(source, tmp_path/'out.wav', 1, 1.25)
    assert sf.info(tmp_path/'out.wav').duration == pytest.approx(1, abs=1/48000)

def test_timeline_preserves_gaps_and_overlapping_speakers(tmp_path):
    storage = LocalStorage(str(tmp_path))
    for key in ('a.wav','b.wav'):
        sf.write(storage.path(key), np.ones((48000,2))*.1,48000,subtype='FLOAT')
    segments = [Segment('a','a',1,2,'A',aligned_audio='a.wav',render_status='translated'),Segment('b','b',1.5,2.5,'B',aligned_audio='b.wav',render_status='translated')]
    media.dialogue_timeline(segments,storage,tmp_path/'timeline.wav',4)
    data, sr = sf.read(tmp_path/'timeline.wav')
    assert len(data) == 4*48000
    assert np.max(np.abs(data[:48000])) == 0
    assert data[60000,0] == pytest.approx(.1, abs=.0001)
    assert data[85000,0] == pytest.approx(.2, abs=.0001)
    assert np.max(np.abs(data[120000:])) == 0

def test_subtitles_keep_absolute_timestamps():
    segment = Segment('a','a',65.123,67.456,'原文',translation='Translation')
    assert '00:01:05,123 --> 00:01:07,456' in media.subtitles([segment],'translation')
    assert media.subtitles([segment],'source_text',True).startswith('WEBVTT\n')

def test_original_fallback_is_not_doubled_and_other_windows_are_replaced(tmp_path):
    storage=LocalStorage(str(tmp_path))
    sf.write(tmp_path/'original.wav',np.ones((4*48000,2))*.1,48000,subtype='FLOAT')
    sf.write(tmp_path/'dub.wav',np.ones((48000,2))*.3,48000,subtype='FLOAT')
    segments=[Segment('bad','a',0,1,'Incomplete',render_status='original'),
              Segment('good','a',2,3,'Translated',aligned_audio='dub.wav',render_status='translated')]
    media.dialogue_timeline(segments,storage,tmp_path/'timeline.wav',4,original=tmp_path/'original.wav')
    data,sr=sf.read(tmp_path/'timeline.wav')
    assert len(data)==4*48000
    assert np.allclose(data[:48000],.1)
    assert np.allclose(data[48000:2*48000],.1)
    assert np.allclose(data[2*48000:3*48000],.3)
    assert np.allclose(data[3*48000:],.1)
