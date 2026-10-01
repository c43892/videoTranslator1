"""Private GPU service. Only media-relative paths cross the container boundary."""
import os
import re
import shutil
import threading
import time
import warnings
import gc
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(os.getenv('STORAGE_ROOT', '/data')).resolve()
MODEL_DIR = Path(os.getenv('MODEL_DIR', '/models/checkpoints-2.5'))
MODEL_REPO = 'IndexTeam/IndexTTS-2.5'
MODEL_REVISION = 'c39ce5ba981572cb187443877ff559dfb246ce63'
STATE = {'status': 'loading', 'detail': 'Downloading IndexTTS2.5 weights', 'model': 'IndexTTS2 2.5.0'}
LOCK = threading.Lock()
MODEL = None
MAX_TOKENS = int(os.getenv('TTS_MAX_TEXT_TOKENS', '120'))


def inference_language(language: str, text: str) -> str:
    """Map the product's language codes to the explicit IndexTTS-2.5 tokens."""
    if language in {'en', 'zh', 'ja', 'es', 'ar'}:
        return language
    if re.search(r'[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]', text):
        return 'ar'
    if re.search(r'[\u3040-\u30ff]', text):
        return 'ja'
    return 'zh' if re.search(r'[\u3400-\u4dbf\u4e00-\u9fff]', text) else 'en'


def media_path(key: str) -> Path:
    path = (ROOT / key).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise HTTPException(400, 'Invalid media key')
    return path


def load_model():
    global MODEL
    try:
        STATE.update(status='loading', detail='Downloading IndexTTS2.5 weights')
        from huggingface_hub import snapshot_download
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(MODEL_REPO, revision=MODEL_REVISION,
                          local_dir=str(MODEL_DIR), max_workers=2,
                          allow_patterns=['*.pth', '*.pt', '*.yaml', '*.tiktoken'])
        STATE['detail'] = 'Loading models onto GPU'
        # Upstream sets HF_HUB_CACHE relative to its cwd; put that cache on our persistent volume.
        os.chdir('/models')
        w2v_dir = MODEL_DIR / 'hf_cache' / 'w2v-bert-2.0'
        w2v_weights = (w2v_dir / 'model.safetensors', w2v_dir / 'pytorch_model.bin')
        if w2v_dir.exists() and not any(path.is_file() for path in w2v_weights):
            # A stopped first boot can leave only metadata and .incomplete files. Upstream checks
            # merely whether the directory exists, so remove that partial directory before retrying.
            shutil.rmtree(w2v_dir)
        STATE['detail'] = 'Downloading IndexTTS2.5 auxiliary models'
        from indextts.utils.model_download import ensure_models_available
        ensure_models_available(str(MODEL_DIR))
        STATE['detail'] = 'Loading models onto GPU'
        from indextts.infer_v2_5 import IndexTTS2
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable inside the TTS container')
        use_bf16 = torch.cuda.is_bf16_supported()
        MODEL = IndexTTS2(cfg_path=str(MODEL_DIR / 'config.yaml'), model_dir=str(MODEL_DIR),
                          device='cuda:0', use_bf16=use_bf16, use_cuda_kernel=False,
                          use_deepspeed=False, use_qwen_emo=False)
        torch.cuda.empty_cache()
        STATE.update(status='ready', detail=torch.cuda.get_device_name(0),
                     precision='bf16' if use_bf16 else 'fp32')
    except Exception as exc:
        import traceback
        traceback.print_exc()
        STATE.update(status='error', detail=f'{type(exc).__name__}: {exc}')


@asynccontextmanager
async def lifespan(app):
    stopped = threading.Event()
    if os.getenv('CLOUD_HEALTH_EXECUTION') == '1':
        from runtime_health import ProcessActivity, gpu_active
        def monitor_activity():
            process = ProcessActivity(os.getpid())
            while not stopped.is_set():
                cpu_active = process.sample()
                if gpu_active() or cpu_active:
                    STATE['activity_seq'] = STATE.get('activity_seq', 0) + 1
                stopped.wait(10)
        threading.Thread(target=monitor_activity, daemon=True).start()
    def warmup():
        with LOCK:load_model()
    threading.Thread(target=warmup, daemon=True).start()
    yield
    stopped.set()


app = FastAPI(lifespan=lifespan)


def unload_model():
    global MODEL
    MODEL = None
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    STATE.update(status='ready',detail='GPU 就绪；IndexTTS2 按需加载')


def ensure_model():
    if MODEL is None:
        load_model()
    if MODEL is None:
        raise HTTPException(503,STATE['detail'])


from demucs_service import install
install(app, LOCK, media_path, unload_model)


@app.get('/health')
def health():
    if STATE['status'] != 'ready':
        from fastapi.responses import JSONResponse
        return JSONResponse(STATE, status_code=503)
    return {**STATE, 'max_text_tokens_per_segment': MAX_TOKENS, 'max_text_characters': 5000,
            'max_reference_seconds': 15, 'languages': ['en', 'zh', 'ja', 'es', 'ar']}


class Texts(BaseModel):
    texts: list[str] = Field(max_length=500)


@app.post('/tokenize')
def tokenize(body: Texts):
    with LOCK:
        ensure_model()
        return {'counts': [len(MODEL.tokenizer.tokenize(t)) for t in body.texts],
                'chunk_limit': MAX_TOKENS}


class Speech(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    language: str = Field(default='auto', pattern='^(auto|en|zh|ja|es|ar)$')
    speaker_audio: str
    emotion_audio: str
    output: str


@app.post('/synthesize')
def synthesize(body: Speech):
    import soundfile as sf
    import numpy as np
    speaker, emotion, output = (media_path(k) for k in (body.speaker_audio, body.emotion_audio, body.output))
    if output.suffix != '.wav':
        raise HTTPException(400, 'Output must be WAV')
    for kind, path in [('speaker', speaker), ('emotion', emotion)]:
        if not path.is_file():
            raise HTTPException(422, f'{kind} reference is missing')
        signal, sr = sf.read(path)
        duration = len(signal) / sr
        if duration <= 0 or not np.isfinite(signal).all():
            raise HTTPException(422, f'Unsuitable {kind} reference: duration or audio quality')
    with LOCK:
        ensure_model()
        count = len(MODEL.tokenizer.tokenize(body.text))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.partial.wav')
        started = time.monotonic()
        try:
            import torch
            torch.manual_seed(42)
            with warnings.catch_warnings():
                # Upstream otherwise saves incomplete speech when its generation cap is hit.
                warnings.filterwarnings('error', message='WARN: generation stopped.*', category=RuntimeWarning)
                for chunk_tokens in dict.fromkeys([MAX_TOKENS, max(20, MAX_TOKENS//2)]):
                    try:
                        MODEL.infer(spk_audio_prompt=str(speaker), emo_audio_prompt=str(emotion), text=body.text,
                                    output_path=str(temporary), lang=inference_language(body.language, body.text),
                                    emo_alpha=1.0, use_random=False,
                                    max_text_tokens_per_segment=chunk_tokens, num_beams=1, verbose=False)
                        break
                    except RuntimeWarning:
                        temporary.unlink(missing_ok=True)
                        if chunk_tokens == max(20, MAX_TOKENS//2):
                            raise
            signal, sr = sf.read(temporary)
            if not len(signal) or not np.isfinite(signal).all() or np.sqrt(np.mean(signal**2)) < 0.0001:
                raise RuntimeError('Synthesized audio is empty or invalid')
            temporary.replace(output)
            return {'output': body.output, 'duration': len(signal) / sr, 'tokens': count,
                    'elapsed_seconds': round(time.monotonic() - started, 2)}
        except RuntimeWarning as exc:
            temporary.unlink(missing_ok=True)
            raise HTTPException(422, 'Incomplete synthesis: shorten this utterance at a natural boundary') from exc
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            import torch
            torch.cuda.empty_cache()
            if 'out of memory' in str(exc).lower():
                raise HTTPException(503, 'GPU memory exhausted; retry after freeing GPU memory or use a larger GPU') from exc
            raise


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)
