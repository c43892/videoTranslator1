import pytest
from videotranslator.adapters import private_engine


@pytest.mark.parametrize('codec,pixel,audio,expected_video,expected_audio', [
    ('h264', 'yuv420p', 'aac', 'copy', 'copy'),
    ('h264', 'yuv420p', 'pcm_s16le', 'copy', 'aac'),
    ('hevc', 'yuv420p', 'aac', 'libx264', 'copy'),
    ('h264', 'yuv420p10le', 'aac', 'libx264', 'copy'),
])
def test_export_preserves_only_browser_compatible_streams(monkeypatch, codec, pixel, audio, expected_video, expected_audio):
    monkeypatch.setattr(private_engine, 'ffprobe_json', lambda _: {'streams': [
        {'codec_type':'video', 'codec_name':codec, 'pix_fmt':pixel},
        {'codec_type':'audio', 'codec_name':audio},
        {'codec_type':'subtitle', 'codec_name':'mov_text'}]})
    args = private_engine.video_export_args('source.mp4')
    assert args[args.index('-c:v') + 1] == expected_video
    assert args[args.index('-c:a') + 1] == expected_audio
    assert '0:s?' in args and 'mov_text' in args and '+faststart' in args
