import subprocess
from types import SimpleNamespace

import pytest
from videotranslator import media


@pytest.mark.parametrize('language,label,tag', [('zh', '你好，世界。', 'zho'), ('en', 'Hello, world.', 'eng')])
def test_embedded_subtitles_preserve_text_timing_and_are_not_burned_in(tmp_path, language, label, tag):
    video, audio, output = [tmp_path / name for name in ('video.mp4', 'audio.wav', 'result.mp4')]
    media.ffmpeg(['-f', 'lavfi', '-i', 'color=s=32x32:d=3', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video])
    media.ffmpeg(['-f', 'lavfi', '-i', 'sine=duration=3', audio])
    subtitle = tmp_path / 'translated.srt'
    subtitle.write_text(media.subtitles([SimpleNamespace(start=.25, end=1.75, translation=label)], 'translation'), encoding='utf-8')
    media.assemble(video, audio, output, media.probe(video), subtitle, language)
    info = media.probe(output)
    stream = next(s for s in info['streams'] if s['codec_type'] == 'subtitle')
    assert stream['codec_name'] == 'mov_text'
    assert stream['tags']['language'] == tag
    assert stream['disposition']['default'] == 1
    text = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output), '-map', '0:s:0', '-f', 'srt', '-']).decode()
    assert label in text
    assert '00:00:00,250 --> 00:00:01,750' in text
    # Identical compressed video stream proves the captions are a separate track.
    def video_hash(path):
        return subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:v:0', '-c', 'copy', '-f', 'hash', '-'])
    assert video_hash(video) == video_hash(output)
    assert float(info['format']['duration']) == pytest.approx(3, abs=.05)


def test_video_without_dialogue_still_assembles(tmp_path):
    video, audio, output = [tmp_path / name for name in ('video.mp4', 'audio.wav', 'result.mp4')]
    media.ffmpeg(['-f', 'lavfi', '-i', 'color=s=32x32:d=1', video])
    media.ffmpeg(['-f', 'lavfi', '-i', 'sine=duration=1', audio])
    media.assemble(video, audio, output, media.probe(video))
    assert {s['codec_type'] for s in media.probe(output)['streams']} == {'audio', 'video'}
