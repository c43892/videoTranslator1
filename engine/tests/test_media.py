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
    assert np.allclose(data[48000:2*48000],0)
    assert np.allclose(data[2*48000+4800:3*48000-4800],.3)
    assert data[2*48000,0] == 0
    assert data[3*48000-1,0] == 0
    assert np.allclose(data[3*48000:],0)


def test_short_fades_remove_join_steps_without_moving_samples(tmp_path):
    storage = LocalStorage(str(tmp_path))
    rate = media.RATE
    sf.write(tmp_path/'original.wav', np.full((3*rate, 2), .2), rate, subtype='FLOAT')
    sf.write(tmp_path/'a.wav', np.full((rate, 2), .6), rate, subtype='FLOAT')
    sf.write(tmp_path/'b.wav', np.full((rate, 2), -.6), rate, subtype='FLOAT')
    segments = [Segment('a','a',.5,1.5,'a',aligned_audio='a.wav',render_status='translated'),
                Segment('b','b',1.5,2.5,'b',aligned_audio='b.wav',render_status='translated')]
    output = tmp_path/'timeline.wav'
    media.dialogue_timeline(segments, storage, output, 3, original=tmp_path/'original.wav')
    data, sr = sf.read(output)
    assert sr == rate and data.shape == (3*rate,2)
    # Legacy close-gap behavior disables the first clip's fade-out. The next
    # clip still uses its 100 ms fade-in, and no original speech fills the join.
    assert np.allclose(data[rate//2+4800:3*rate//2], .6)
    assert np.allclose(data[3*rate//2+4800:5*rate//2-4800], -.6)
    assert data[3*rate//2-1,0] == pytest.approx(.6)
    assert data[3*rate//2,0] == 0
    assert np.allclose(data[:rate//2], 0)


def test_fades_follow_sound_inside_digital_padding(tmp_path):
    storage = LocalStorage(str(tmp_path))
    rate = media.RATE
    signal = np.zeros((rate,2), dtype=np.float32)
    signal[1000:12000] = .5
    sf.write(tmp_path/'dub.wav', signal, rate, subtype='FLOAT')
    segment = Segment('a','a',0,1,'a',aligned_audio='dub.wav',render_status='translated')
    media.dialogue_timeline([segment], storage, tmp_path/'timeline.wav', 1)
    data, _ = sf.read(tmp_path/'timeline.wav')
    assert data[1000,0] == data[11999,0] == 0
    assert np.max(data[:,0]) > .49
    assert np.max(np.abs(np.diff(data[:,0]))) < .001
    assert not np.any(data[12000:])


@pytest.mark.parametrize('length', [1,2,3,10,240])
def test_very_short_clips_have_finite_bounded_fades(tmp_path,length):
    storage = LocalStorage(str(tmp_path))
    sf.write(tmp_path/'dub.wav', np.full((length,2),.5), media.RATE, subtype='FLOAT')
    segment = Segment('a','a',0,length/media.RATE,'a',aligned_audio='dub.wav',render_status='translated')
    media.dialogue_timeline([segment],storage,tmp_path/'timeline.wav',length/media.RATE)
    data,_ = sf.read(tmp_path/'timeline.wav')
    assert len(data)==length and np.all(np.isfinite(data))
    assert np.max(np.abs(data))<=.5 and data[0,0]==data[-1,0]==0


def test_preserved_event_overlaps_are_not_doubled_or_hard_cut(tmp_path):
    storage=LocalStorage(str(tmp_path));rate=media.RATE
    sf.write(tmp_path/'original.wav',np.full((2*rate,2),.2),rate,subtype='FLOAT')
    sf.write(tmp_path/'dub.wav',np.full((2*rate,2),.1),rate,subtype='FLOAT')
    segment=Segment('a','a',0,2,'a',aligned_audio='dub.wav',render_status='translated')
    media.dialogue_timeline([segment],storage,tmp_path/'timeline.wav',2,original=tmp_path/'original.wav',
        preserve_intervals=[{'start':.5,'end':1.2},{'start':1,'end':1.5}])
    data,_=sf.read(tmp_path/'timeline.wav')
    assert np.allclose(data[rate//2+4800:3*rate//2-4800],.3)
    assert np.max(np.abs(np.diff(data[:,0])))<.005
