"""Real audio regression: duration compression retains pitch and the end of a clip."""
import numpy as np
import pytest
import soundfile as sf
from videotranslator import media


def peak(signal, rate):
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    return np.fft.rfftfreq(len(signal), 1/rate)[np.argmax(spectrum)]


@pytest.mark.parametrize('duration,window', [(4.36535,3.38),(4,1),(9,1)])
def test_compression_preserves_pitch_and_tail(tmp_path,duration,window):
    rate = 48000
    # A different final tone detects truncation of the end instead of compression.
    n = round(duration*rate); t = np.arange(n)/rate
    signal = .2*np.sin(2*np.pi*440*t)
    signal[round(n*.7):] = .2*np.sin(2*np.pi*880*t[round(n*.7):])
    source,out = tmp_path/'source.wav',tmp_path/'out.wav'
    sf.write(source,signal,rate)
    media.align(source,out,window,1.25)
    audio,sr = sf.read(out,always_2d=True); mono = audio.mean(axis=1)
    assert sr == rate and len(audio) == round(window*rate)
    assert peak(mono[round(len(mono)*.15):round(len(mono)*.4)],sr) == pytest.approx(440,abs=8)
    assert peak(mono[round(len(mono)*.78):round(len(mono)*.9)],sr) == pytest.approx(880,abs=12)


@pytest.mark.parametrize('window',[0,-1,float('nan'),float('inf')])
def test_invalid_window_is_not_silently_accepted(tmp_path,window):
    source = tmp_path/'source.wav'
    sf.write(source,np.ones(48000)*.1,48000)
    with pytest.raises(ValueError):
        media.align(source,tmp_path/'out.wav',window,1.25)
