"""Pin auxiliary weights to the revisions present in the validated local model cache."""
from pathlib import Path

path = Path('/opt/index-tts/indextts/infer_v2.py')
source = path.read_text()
replacements = {
    'from_pretrained("facebook/w2v-bert-2.0")':
        'from_pretrained("facebook/w2v-bert-2.0", revision="da985ba0987f70aaeb84a80f2851cfac8c697a7b")',
    'hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")':
        'hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors", revision="265c6cef07625665d0c28d2faafb1415562379dc")',
    '"funasr/campplus", filename="campplus_cn_common.bin"':
        '"funasr/campplus", filename="campplus_cn_common.bin", revision="e4b6ede7ce16997aff4ae69fbca1f0175e2afede"',
    'from_pretrained(bigvgan_name, use_cuda_kernel=self.use_cuda_kernel)':
        'from_pretrained(bigvgan_name, revision="633ff708ed5b74903e86ff1298cf4a98e921c513", use_cuda_kernel=self.use_cuda_kernel)',
}
for old, new in replacements.items():
    assert source.count(old) == 1, 'Upstream model loader changed; review pinning'
    source = source.replace(old, new)
path.write_text(source)
