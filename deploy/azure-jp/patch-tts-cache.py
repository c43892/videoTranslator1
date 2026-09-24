"""Relocate upstream text-normalization caches for a non-root runtime."""
from pathlib import Path

path = Path('/opt/index-tts/indextts/utils/front.py')
source = path.read_text()
old = 'cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tagger_cache")'
new = 'cache_dir = os.environ.get("TTS_NORMALIZER_CACHE", "/tmp/tts-normalizer")'
assert source.count(old) == 1, 'Upstream normalizer changed; review before patching'
assert source.count('NormalizerEn(overwrite_cache=False)') == 1
source = source.replace(old, new).replace('NormalizerEn(overwrite_cache=False)',
                                        'NormalizerEn(cache_dir=cache_dir, overwrite_cache=False)')
path.write_text(source)
